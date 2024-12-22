import datetime
import lakeapi
import sqlite3
import pandas as pd
import os
import gc
import json
import boto3
import argparse



def download_data(table, symbol, exchange, start_date, end_date, aws_session):
    """Download data from lakeapi based on provided parameters."""
    print(f"Downloading data for {symbol} from {exchange} between {start_date} and {end_date}")
    return lakeapi.load_data(
        table=table,
        start=start_date,
        end=end_date,
        symbols=[symbol],
        exchanges=[exchange],
        boto3_session=aws_session
    )

def remove_time_anomalies(books):
    """Remove rows with time anomalies based on sequence_number and time columns."""
    print("Removing time anomalies...")
    books_sorted = books.sort_values(by='sequence_number').reset_index(drop=True)
    anomalies_origin_time = (books_sorted['sequence_number'] > books_sorted['sequence_number'].shift(1)) & \
                            (books_sorted['origin_time'] < books_sorted['origin_time'].shift(1))
    anomalies_received_time = (books_sorted['sequence_number'] > books_sorted['sequence_number'].shift(1)) & \
                              (books_sorted['received_time'] < books_sorted['received_time'].shift(1))
    anomalies = anomalies_origin_time | anomalies_received_time
    books_filtered = books_sorted[~anomalies].reset_index(drop=True)
    print(f"Removed {anomalies.sum()} time anomalies.")
    return books_filtered

def filter_price_anomalies(books):
    """Filter rows where ask price is greater than bid price."""
    print("Filtering price anomalies (ask price <= bid price)...")
    len_book_before = len(books)
    books = books.loc[books['ask_0_price'] > books['bid_0_price']]
    print(f"Filtered {len_book_before - len(books)} rows with price anomalies.")
    return books

def reorder_columns(books):
    """Reorder columns and filter bid/ask columns up to level 10."""
    print("Reordering columns and filtering for bid/ask up to level 10...")
    static_columns = ['origin_time', 'received_time', 'sequence_number', 'symbol', 'exchange']
    bid_columns = [f'bid_{i}_price' for i in range(10)] + [f'bid_{i}_size' for i in range(10)]
    ask_columns = [f'ask_{i}_price' for i in range(10)] + [f'ask_{i}_size' for i in range(10)]
    selected_columns = static_columns + bid_columns + ask_columns
    books = books[selected_columns]
    print(f"Reordered columns, final shape: {books.shape}")
    return books

def save_to_sqlite(books, table, path_sqlite, date):
    """Save the cleaned data to an SQLite database in the specified path."""
    db_name = os.path.join(path_sqlite, f"{table}.sqlite")
    conn = sqlite3.connect(db_name)
    conn.execute('PRAGMA journal_mode=WAL;')
    cursor = conn.cursor()
    
    # Add a 'date' column for the date extracted from 'origin_time'
    books['date'] = books['origin_time'].dt.date
    books.to_sql(table, conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print(f"Data saved to SQLite at {db_name}.")

def process_data(table, symbol, exchange, start_date, end_date, aws_session, path_sqlite):
    """Main function to process data for each day and store it in the SQLite database."""
    current_date = start_date
    while current_date <= end_date:
        print(f"\n--- Processing date: {current_date} ---")
        next_date = current_date + datetime.timedelta(days=1)
        books = download_data(table, symbol, exchange, current_date, next_date, aws_session)

        if not books.empty:
            books = reorder_columns(books)
            books = filter_price_anomalies(books)
            books = remove_time_anomalies(books)
            save_to_sqlite(books, table, path_sqlite, current_date)
            del books
            gc.collect()
        else:
            print(f"No data available for date: {current_date}")

        current_date = next_date



def load_aws_credentials(file_path):
    """
    Load AWS credentials from a text file.

    Args:
        file_path (str): Path to the credentials file.

    Returns:
        dict: A dictionary with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
    """
    credentials = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() and "=" in line:
                    key, value = line.strip().split('=', 1)
                    credentials[key] = value
    except FileNotFoundError:
        raise FileNotFoundError(f"Credentials file not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading credentials file: {e}")
    
    if not {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}.issubset(credentials):
        raise ValueError("Invalid credentials file. Missing required keys.")
    
    return credentials

# Load the credentials
credentials_path = "crypto-lake_aws_credentials.txt"
aws_credentials = load_aws_credentials(credentials_path)

# Create an AWS session using the credentials

aws_session = boto3.Session(
    aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
    region_name=aws_credentials["region_name"]
)





if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Download LOB book data and fill sqlite database.")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol for the pair, e.g., BTC-USDT.")
    parser.add_argument("--startdate", type=str, required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--enddate", type=str, required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    table = config["table"]
    exchange = config["exchanges"]
    path_sqlite = config["path_sqlite"]

    # Parse dates
    start_date = datetime.datetime.strptime(args.startdate, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(args.enddate, "%Y-%m-%d")

    # Process data
    process_data(
        table=table,
        symbol=args.symbol,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        aws_session=aws_session,
        path_sqlite=path_sqlite
    )
