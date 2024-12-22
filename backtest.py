import pandas as pd
import os
import time
import json
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef, classification_report
import seaborn as sns
import zipfile
import pickle
import gzip
import gc
import random
import sqlite3
from tqdm import tqdm
from collections import defaultdict
import torchmetrics
import pytorch_lightning
from utils_pipeline import *



def format_xticks(ax, max_value):
    scale = 1e6 if max_value >= 1e6 else 1e5
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / scale:.2f}"))
    ax.annotate(f"x {int(scale):.0e}", xy=(1, 0), xycoords='axes fraction', 
                fontsize=8, color='dimgray', ha='right', va='bottom')

def scientific_formatter(y, pos):
    if abs(y) < 0.01 or abs(y) > 1e2:
        return f"{y:.1e}"
    else:
        return f"{y:.2f}"

def neg_log_formatter(x, pos):
    return f"-{mticker.LogFormatterSciNotation()(x, pos)}"


def convert_to_datetime(the_serie):
    return pd.to_datetime(the_serie.apply(lambda x: str(x).split('.')[0]))



def fetch_data_between_dates(table, symbols, exchanges, start_date, end_date, path_sqlite, columns=[]):
    """
    Fetch data between start_date and end_date for the specified symbols and exchanges,
    with an option to specify a subset of bid and ask columns to reduce memory usage.

    Parameters:
    - table: The name of the table in the SQLite database.
    - symbols: List of symbols to query.
    - exchanges: List of exchanges to query.
    - start_date: The start date for the data.
    - end_date: The end date for the data.
    - path_sqlite: Path to the SQLite database.
    - columns: List of specific bid and ask columns to retrieve (default is empty, meaning retrieve all columns).

    Returns:
    - A DataFrame containing the requested data.
    """

    # Default columns to always retrieve
    base_columns = ['date', 'origin_time', 'received_time', 'sequence_number', 'symbol', 'exchange']

    # If specific columns are passed, retrieve those along with the base columns
    if columns:
        selected_columns = base_columns + columns
        column_query = ', '.join(selected_columns)
    else:
        # Retrieve all columns if no specific ones are passed
        column_query = '*'

    # Construct the database path
    db_name = os.path.join(path_sqlite, f"{table}.sqlite")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Prepare an empty list to store data
    all_data = []

    # Iterate over each symbol and exchange
    for symbol in symbols:
        for exchange in exchanges:
            # SQL query to fetch data between the specified dates for the given symbol and exchange
            query = f"""
            SELECT {column_query}
            FROM {table}
            WHERE date BETWEEN ? AND ?
            AND symbol = ?
            AND exchange = ?
            """

            # Execute the query and fetch the result
            result = pd.read_sql_query(query, conn, params=(start_date, end_date, symbol, exchange))

            # Check if any data was returned
            if result.empty:
                print(f"No data found for {symbol} on {exchange} between {start_date} and {end_date}.")
            else:
                print(f"Data found for {symbol} on {exchange} between {start_date} and {end_date}.")
                all_data.append(result)

    # Close the connection
    conn.close()

    # Combine all the data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

        combined_data['origin_time_dt'] = convert_to_datetime(combined_data['origin_time'])
        combined_data['received_time_dt'] = convert_to_datetime(combined_data['received_time'])

        return combined_data
    else:
        return None




def get_dates(table, symbols, exchanges, start_date, end_date, path_sqlite, min_rows=5000):
    """
    Fetch unique dates with at least `min_rows` where 'bid_0_price' is not null between start_date and end_date.

    Parameters:
    - table: The name of the table in the SQLite database.
    - symbols: List of symbols to query.
    - exchanges: List of exchanges to query.
    - start_date: The start date for the data.
    - end_date: The end date for the data.
    - path_sqlite: Path to the SQLite database.
    - min_rows: The minimum number of rows required with a non-null 'bid_0_price' column.

    Returns:
    - A list of dates that meet the minimum row criteria.
    """

    # Construct the database path
    db_name = os.path.join(path_sqlite, f"{table}.sqlite")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    dict_dates = {}

    # Iterate over each symbol and exchange
    for symbol in symbols:

        # Placeholder for all dates with sufficient data
        valid_dates = []

        for exchange in exchanges:
            # SQL query to count rows per date where 'bid_0_price' is not null
            query = f"""
            SELECT date, COUNT(*) as row_count
            FROM {table}
            WHERE date BETWEEN ? AND ?
            AND symbol = ?
            AND exchange = ?
            AND bid_0_price IS NOT NULL
            GROUP BY date
            HAVING row_count >= ?
            """

            # Execute the query and fetch results
            result = pd.read_sql_query(query, conn, params=(start_date, end_date, symbol, exchange, min_rows))

            # Add valid dates to the list
            if not result.empty:
                valid_dates += result['date'].tolist()

        dict_dates[symbol] = sorted(valid_dates)

    # Close the database connection
    conn.close()

    # Return the list of dates
    return dict_dates





def split_dates(dates, train_ratio=0.6, val_ratio=0.2):
    """
    Split a list of dates into train, validation, and test sets.

    Parameters:
    - dates (list): A list of available dates.
    - train_ratio (float): Proportion of data for training.
    - val_ratio (float): Proportion of data for validation.

    Returns:
    - train_dates, val_dates, test_dates: Lists of dates for each split.
    """
    n_total_dates = len(dates)

    n_train = int(n_total_dates * train_ratio)
    n_val = int(n_total_dates * val_ratio)

    # Split the dates into train, validation, and test sets
    train_dates = dates[:n_train]
    val_dates = dates[n_train:n_train + n_val]
    test_dates = dates[n_train + n_val:]

    # Output the date splits
    print("Training Dates:", train_dates[0], '..', train_dates[-1], ' : ', len(train_dates), ' dates' )
    print("Validation Dates:", val_dates[0], '..', val_dates[-1], ' : ', len(val_dates), ' dates' )
    print("Test Dates:", test_dates[0], '..', test_dates[-1], ' : ', len(test_dates), ' dates' )

    return train_dates, val_dates, test_dates






def __get_fees_free_pnl__(trading_simulation):
    df = trading_simulation
    profit_list = []
    for index, row in df.iterrows():
        profit_no_fees = 0
        if row.Type == 'Long':
            local_profit = (row.Price_Exit_Long - row.Price_Entry_Long)
            profit_no_fees += local_profit
        elif row.Type == 'Short':
            local_profit = (row.Price_Entry_Short - row.Price_Exit_Short)
            profit_no_fees += local_profit

        profit_list.append(profit_no_fees)
    return profit_list


def __get_pnl_with_fees__(trading_simulation, trading_hyperparameters):
    df = trading_simulation
    profit_list = []
    for index, row in df.iterrows():
        profit_no_fees = 0
        if row.Type == 'Long':
            local_profit = (row.Price_Exit_Long - row.Price_Entry_Long) - (row.Price_Exit_Long * trading_hyperparameters['trading_fee']) - (row.Price_Entry_Long * trading_hyperparameters['trading_fee'])
            profit_no_fees += local_profit
        elif row.Type == 'Short':
            local_profit = (row.Price_Entry_Short - row.Price_Exit_Short) - (row.Price_Entry_Short * trading_hyperparameters['trading_fee']) - (row.Price_Exit_Short * trading_hyperparameters['trading_fee'])
            profit_no_fees += local_profit

        profit_list.append(profit_no_fees)
    return profit_list


def __get_long_short_indices__(trading_simulation):
    long_indices = []
    short_indices = []
    for index, row in trading_simulation.iterrows():
        if row.Type == 'Long':
            long_indices.append(pd.to_datetime(row.Entry_Long))
        elif row.Type == 'Short':
            short_indices.append(pd.to_datetime(row.Entry_Short))

    return long_indices, short_indices


def __get_fees_free_returns__(trading_simulation):
    df = trading_simulation
    returns_list = []
    for index, row in df.iterrows():
        trade_return = 0
        if row.Type == 'Long':
            # Calculate return for long trades
            trade_return = (row.Price_Exit_Long - row.Price_Entry_Long) / row.Price_Entry_Long
        elif row.Type == 'Short':
            # Calculate return for short trades
            trade_return = (row.Price_Entry_Short - row.Price_Exit_Short) / row.Price_Entry_Short

        returns_list.append(trade_return)
    return returns_list


def __get_fees_free_log_returns__(trading_simulation):
    df = trading_simulation.copy(deep = True)
    log_returns_list = []
    for index, row in df.iterrows():
        log_return = 0
        if row.Type == 'Long':
            # Calculate log return for long trades
            log_return = np.log(row.Price_Exit_Long / row.Price_Entry_Long)
        elif row.Type == 'Short':
            # Calculate log return for short trades
            log_return = np.log(row.Price_Entry_Short / row.Price_Exit_Short)

        log_returns_list.append(log_return)

    df['log_returns'] = log_returns_list

    return df


def __get_time_difference__(trading_simulation):
    df = trading_simulation.copy(deep = True)
    td_list = []
    for index, row in df.iterrows():
        td = 0
        if row.Type == 'Long':
            # Calculate log return for long trades
            td = (row.Exit_Long - row.Entry_Long).seconds
        elif row.Type == 'Short':
            # Calculate log return for short trades
            td = (row.Exit_Short - row.Entry_Short).seconds

        td_list.append(td)

    df['duration_trade_s'] = td_list

    return df



def merge_trading_history(prices, trading_history_dataframe):
    # Prepare trading history DataFrame by reshaping for merging

    entry_short = trading_history_dataframe2[['Exit_Short_Sequence', 'Entry_Short', 'log_returns', 'duration_trade_s']].dropna()
    entry_short.columns = ['sequence_number', 'Action', 'log_returns', 'duration_trade_s']
    entry_short['Action'] = 'Short'

    entry_long = trading_history_dataframe2[['Exit_Long_Sequence', 'Entry_Long', 'log_returns', 'duration_trade_s']].dropna()
    entry_long.columns = ['sequence_number', 'Action', 'log_returns', 'duration_trade_s']
    entry_long['Action'] = 'Long'

    # Concatenate all actions
    merged_trades = pd.concat([entry_short, entry_long], ignore_index=True).sort_values('sequence_number')
    merged_prices = pd.merge(prices, merged_trades, on='sequence_number', how='left')

    return merged_prices



# Define function to handle formats
def safe_parse_date(x):
    try:
        return pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
        except ValueError:
            return pd.NaT


def fetch_small_data_between_dates(table, symbols, exchanges, start_date, end_date, path_sqlite):
    """
    Fetch data between start_date and end_date for the specified symbols and exchanges,
    retrieving specific columns including bid and ask prices and volumes.

    Parameters:
    - table: The name of the table in the SQLite database.
    - symbols: List of symbols to query.
    - exchanges: List of exchanges to query.
    - start_date: The start date for the data.
    - end_date: The end date for the data.
    - path_sqlite: Path to the SQLite database.

    Returns:
    - A DataFrame containing the requested data.
    """

    # Default columns to always retrieve
    base_columns = ['date', 'origin_time', 'received_time', 'sequence_number', 'symbol', 'exchange']

    # Specific bid and ask columns
    bid_ask_columns = ['bid_0_price', 'bid_1_price', 'ask_0_price', 'ask_1_price', 'bid_0_size', 'bid_1_size', 'ask_0_size', 'ask_1_size']

    # Combine base columns and bid/ask columns
    selected_columns = base_columns + bid_ask_columns
    column_query = ', '.join(selected_columns)

    # Construct the database path
    db_name = os.path.join(path_sqlite, f"{table}.sqlite")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)

    # Prepare an empty list to store data
    all_data = []

    # Iterate over each symbol and exchange
    for symbol in symbols:
        for exchange in exchanges:
            # SQL query to fetch data between the specified dates for the given symbol and exchange
            query = f"""
            SELECT {column_query}
            FROM {table}
            WHERE date BETWEEN ? AND ?
            AND symbol = ?
            AND exchange = ?
            """

            # Execute the query and fetch the result
            result = pd.read_sql_query(query, conn, params=(start_date, end_date, symbol, exchange))

            # Check if any data was returned
            if result.empty:
                print(f"No data found for {symbol} on {exchange} between {start_date} and {end_date}.")
            else:
            #    print(f"Data found for {symbol} on {exchange} between {start_date} and {end_date}.")
                all_data.append(result)

    # Close the connection
    conn.close()

    # Combine all the data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

        #combined_data['seconds'] = convert_to_datetime(combined_data['origin_time'])
        combined_data['origin_dt'] = combined_data['origin_time'].apply(safe_parse_date)
        #combined_data['received_time_dt'] = convert_to_datetime(combined_data['received_time'])
        combined_data['received_dt'] = combined_data['received_time'].apply(safe_parse_date)

        return combined_data
    else:
        return None



def check_table_columns(table, path_sqlite):
    """
    Check the column names of a table in an SQLite database.

    Parameters:
    - table: The name of the table in the SQLite database.
    - path_sqlite: Path to the SQLite database.

    Returns:
    - A list of column names in the table.
    """

    # Construct the database path
    db_name = os.path.join(path_sqlite, f"{table}.sqlite")

    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Query to get the column names
    cursor.execute(f"PRAGMA table_info({table})")

    # Fetch all column names
    columns_info = cursor.fetchall()
    column_names = [column[1] for column in columns_info]

    # Close the connection
    conn.close()

    return column_names



def get_backtest_data(table, test_dates, symbols, exchanges, path_sqlite):

    test_data_list = []
    for current_date in test_dates:
        print(f"Processing data for {current_date}")

        # Fetch the current day's data
        data_current = fetch_small_data_between_dates(table, symbols, exchanges, current_date, current_date, path_sqlite)
        data_current['Mid'] = 0.5*(data_current['bid_0_price'] + data_current['ask_0_price'])
        data_current['BIDp1'] = data_current['bid_0_price']
        data_current['ASKp1'] = data_current['ask_0_price']
        data_current = data_current.drop(columns = ['origin_time','received_time','exchange','symbol'])
        test_data_list.append(data_current)

    combined_data = pd.concat(test_data_list)

    return combined_data



class Trading:
    def __init__(self):
        self.long_inventory = 0
        self.short_inventory = 0
        self.long_price = 0
        self.short_price = 0
        self.sequence_number_entry_long = 0
        self.sequence_number_entry_short = 0
        #self.sequence_number_exit_long = 0
        #self.sequence_number_exit_short = 0
        self.date_time_entry_long = None
        #self.date_time_exit_long = None
        self.date_time_entry_short = None
        #self.date_time_exit_short = None
        self.trading_history = []

    def long(self, price, datetime=None, sequence_number=None):
        amount = 1
        self.long_inventory += amount
        self.long_price = price
        self.date_time_entry_long = datetime
        self.sequence_number_entry_long = sequence_number

    def short(self, price, datetime=None, sequence_number=None):
        amount = 1
        self.short_inventory += amount
        self.short_price = price
        self.date_time_entry_short = datetime
        self.sequence_number_entry_short = sequence_number

    def exit_long(self, price, datetime=None, sequence_number=None):
        self.trading_history.append({'Type': 'Long',
                                     'Entry_Long_Sequence':self.sequence_number_entry_long,
                                     'Entry_Long': self.date_time_entry_long,
                                     'Price_Entry_Long': self.long_price,
                                     'Exit_Long': datetime,
                                     'Exit_Long_Sequence':sequence_number,
                                     'Price_Exit_Long': price})
        self.long_inventory = 0
        self.long_price = 0
        self.date_time_entry_long = None
        self.sequence_number_entry_long = None

    def exit_short(self, price, datetime=None, sequence_number=None):
        self.trading_history.append({'Type': 'Short',
                                     'Entry_Short_Sequence':self.sequence_number_entry_short,
                                     'Entry_Short': self.date_time_entry_short,
                                     'Price_Entry_Short': self.short_price,
                                     'Exit_Short': datetime,
                                     'Exit_Short_Sequence':sequence_number,
                                     'Price_Exit_Short': price})
        self.short_inventory = 0
        self.short_price = 0
        self.date_time_entry_short = None
        self.sequence_number_entry_short = None






def backtest(prices, trading_hyperparameters):

    TradingAgent = Trading()

    #preds = np.array(test_results['outputs'])
    #probabilities = np.array(test_results['probabilities'])
    probabilities = prices[['prob_0', 'prob_1', 'prob_2']].to_numpy()

    prices.reset_index(drop=True, inplace=True)
    indices_to_delete = prices[prices['prediction'] == 1].index
    prices = prices.drop(indices_to_delete)
    mask = (prices['prediction'] != prices['prediction'].shift()) | (prices.index == 0)
    prices = prices[mask]
    prices = prices.reset_index(drop=True)
    predictions = prices['prediction'].tolist()
    prices = prices.drop(columns=['prediction'])
    prices.reset_index(drop=True, inplace=True)

    for i in tqdm(range(len(prices))):
        mid_price = prices.at[i, "Mid"]
        best_bid_price = prices.at[i, "BIDp1"]
        best_ask_price = prices.at[i, "ASKp1"]
        timestamp = prices.at[i, "seconds"]
        sequence_number = prices.at[i, "sequence_number"]
        prediction = predictions[i]
        probability = np.max(probabilities[i])

        if trading_hyperparameters['mid_side_trading'] == 'mid_to_mid':
                if prediction == 2 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.long(mid_price, timestamp, sequence_number)
                    elif TradingAgent.long_inventory == 0 and TradingAgent.short_inventory > 0:
                        TradingAgent.exit_short(mid_price, timestamp, sequence_number)
                        TradingAgent.long(mid_price, timestamp, sequence_number)
                elif prediction == 0 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.short(mid_price, timestamp, sequence_number)
                    elif TradingAgent.short_inventory == 0 and TradingAgent.long_inventory > 0:
                        TradingAgent.exit_long(mid_price, timestamp, sequence_number)
                        TradingAgent.short(mid_price, timestamp, sequence_number)
        elif trading_hyperparameters['mid_side_trading'] == 'side_market_orders':
                if prediction == 2 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.long(best_ask_price, timestamp, sequence_number)
                    elif TradingAgent.long_inventory == 0 and TradingAgent.short_inventory > 0:
                        TradingAgent.exit_short(best_ask_price, timestamp, sequence_number)
                        TradingAgent.long(best_ask_price, timestamp, sequence_number)
                elif prediction == 0 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.short(best_bid_price, timestamp, sequence_number)
                    elif TradingAgent.short_inventory == 0 and TradingAgent.long_inventory > 0:
                        TradingAgent.exit_long(best_bid_price, timestamp, sequence_number)
                        TradingAgent.short(best_bid_price, timestamp, sequence_number)
        elif trading_hyperparameters['mid_side_trading'] == 'side_limit_orders':
                if prediction == 2 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.long(best_bid_price, timestamp, sequence_number)
                    elif TradingAgent.long_inventory == 0 and TradingAgent.short_inventory > 0:
                        TradingAgent.exit_short(best_bid_price, timestamp, sequence_number)
                        TradingAgent.long(best_bid_price, timestamp, sequence_number)
                elif prediction == 0 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.short(best_ask_price, timestamp, sequence_number)
                    elif TradingAgent.short_inventory == 0 and TradingAgent.long_inventory > 0:
                        TradingAgent.exit_long(best_ask_price, timestamp, sequence_number)
                        TradingAgent.short(best_ask_price, timestamp, sequence_number)

    trading_history_dataframe = pd.DataFrame(TradingAgent.trading_history)

    return trading_history_dataframe



def merge_trading_history(prices, trading_history_dataframe):
    # Prepare trading history DataFrame by reshaping for merging

    entry_short = trading_history_dataframe[['Exit_Short_Sequence', 'Entry_Short', 'log_returns', 'duration_trade_s']].dropna()
    entry_short.columns = ['sequence_number', 'Action', 'log_returns', 'duration_trade_s']
    entry_short['Action'] = 'Short'

    entry_long = trading_history_dataframe[['Exit_Long_Sequence', 'Entry_Long', 'log_returns', 'duration_trade_s']].dropna()
    entry_long.columns = ['sequence_number', 'Action', 'log_returns', 'duration_trade_s']
    entry_long['Action'] = 'Long'

    # Concatenate all actions
    merged_trades = pd.concat([entry_short, entry_long], ignore_index=True).sort_values('sequence_number')
    merged_prices = pd.merge(prices[['date','sequence_number','seconds','Mid']], merged_trades, on='sequence_number', how='left')

    return merged_prices



def backtest2(df, trading_hyperparameters):

    #df = prices_model.copy(deep=True)

    probabilities = df[['prob_0', 'prob_1', 'prob_2']].to_numpy()
    probas = np.max(probabilities, axis = 1)

    df['Action'] = None
    df['Position'] = None
    df['position_nb'] = np.nan
    df['nb_trades'] = 0

    # Define conditions for actions
    long_condition = (df['prediction'] == 2)
    short_condition = (df['prediction'] == 0)

    # Compute actions using vectorized logic
    df.loc[long_condition, 'Position'] = 'Long'
    df.loc[short_condition, 'Position'] = 'Short'

    df['Position'] = df['Position'].ffill()

    df['Trade'] = df['Position'].ne(df['Position'].shift())
    df.loc[df['Trade'], 'Action'] = df.loc[df['Trade'], 'Position']
    df['Action'] = df['Action'].fillna('Hold')

    df['PnL_gross'] = np.where(df['Position'].shift(1) == 'Long', df['ret'], -df['ret'])
    df['PnL_gross'] = df['PnL_gross'].fillna(0.0)

    df.loc[df['Trade'],'position_nb'] = df.loc[df['Trade'],'Trade'].cumsum()

    df['nb_trades'] = 2*(df['position_nb'] > 0)
    df.loc[df['position_nb'] == 1, 'nb_trades'] = 1

    df['position_nb'] = df['position_nb'].ffill()
    df['position_nb'] = df['position_nb'].fillna(0)

    return df



def get_trades_history(prices_model, trading_hyperparameters):

    trading_history_dataframe = backtest(
        prices_model.copy(deep=True), trading_hyperparameters
    )

    pnl_history = __get_fees_free_pnl__(trading_history_dataframe)
    long_indices_history, short_indices_history = __get_long_short_indices__(trading_history_dataframe)
    trading_history_dataframe2 = __get_fees_free_log_returns__(trading_history_dataframe)
    trading_history_dataframe2 = __get_time_difference__(trading_history_dataframe2)
    #merged_prices = merge_trading_history(prices_model, trading_history_dataframe2)

    log_pnl_history = trading_history_dataframe2['log_returns']
    duration_trade = trading_history_dataframe2['duration_trade_s']

    return log_pnl_history, duration_trade



def get_trades_history2(prices_model, trading_hyperparameters):

    trading_history_dataframe = backtest2(prices_model, trading_hyperparameters)

    log_pnl_history = trading_history_dataframe['PnL']
    duration_trade = trading_history_dataframe['time_diff']

    return log_pnl_history, duration_trade



def sharpe_ratio(returns, risk_free_rate=0):
    sharpe_ratio = (np.mean(returns) - risk_free_rate) / np.std(returns)
    return sharpe_ratio


def sortino_ratio(returns, risk_free_rate=0):
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns)
    excess_returns = np.mean(returns - risk_free_rate)
    return excess_returns / downside_deviation

def maximum_drawdown_log_returns(log_returns):
    cumulative_returns = np.exp(np.cumsum(log_returns))  # Convert log-returns to cumulative geometric returns
    peak = np.maximum.accumulate(cumulative_returns)
    valid_peaks = peak > 0
    drawdown = np.zeros_like(cumulative_returns)
    drawdown[valid_peaks] = (cumulative_returns[valid_peaks] - peak[valid_peaks]) / peak[valid_peaks]
    return drawdown.min()

def maximum_drawdown_geometric_returns(log_returns):
    cumulative_returns = np.cumsum(np.exp(log_returns)-1)  # Convert log-returns to cumulative geometric returns
    peak = np.maximum.accumulate(cumulative_returns)
    valid_peaks = peak > 0
    drawdown = np.zeros_like(cumulative_returns)
    drawdown[valid_peaks] = (cumulative_returns[valid_peaks] - peak[valid_peaks]) / peak[valid_peaks]
    return drawdown.min()

def win_loss_ratio(returns):
    wins = np.sum(returns > 0)
    losses = np.sum(returns < 0)
    return wins / losses if losses > 0 else float('inf')


def value_at_risk(returns, confidence_level=0.05):
    return np.percentile(returns, confidence_level * 100)


def expected_shortfall(returns, confidence_level=0.05):
    var_threshold = value_at_risk(returns, confidence_level)
    return np.mean(returns[returns <= var_threshold])


def cumulative_log_return(log_returns):
    total_log_return = np.sum(log_returns)
    return np.exp(total_log_return) - 1


def annualized_log_return(log_returns, frequency='seconds'):
    # Sum log-returns to get the total log return
    total_log_return = np.sum(log_returns)

    # Number of periods in one year (31,536,000 seconds in a year)
    periods_per_year = 31_536_000  # Assuming frequency in seconds

    # Convert to annualized log return
    annual_log_return = total_log_return * (periods_per_year / len(log_returns))
    return np.exp(annual_log_return) - 1


def average_trade_duration(pnl):
    return len(pnl)  # Assuming each PnL point represents one trade, length = number of trades


def volatility(pnl, trading_periods):
    return np.std(pnl, ddof=1) * np.sqrt(trading_periods)




# prediction changes vs threshold
def make_prediction(row, threshold):
    # Find the maximum of the three probabilities
    max_prob = max(row['prob_0'], row['prob_1'], row['prob_2'])

    # Check which probability is the maximum and apply the threshold logic
    if max_prob == row['prob_0'] and row['prob_0'] > threshold:
        return 0
    elif max_prob == row['prob_2'] and row['prob_2'] > threshold:
        return 2
    else:
        # Default to class 1 if neither prob_0 nor prob_2 exceeds the threshold
        return 1



def compute_pnl_distribution_by_class(trading_history_dataframe, class_0_pnl, class_2_pnl):
    # Filter trades for Class 0 (short positions) and Class 2 (long positions)
    class_0_trades = trading_history_dataframe[trading_history_dataframe['Type'] == 'Short']
    class_2_trades = trading_history_dataframe[trading_history_dataframe['Type'] == 'Long']

    # Compute PnL for class 0 and class 2 using __get_fees_free_pnl__ function
    class_0_pnl = __get_fees_free_pnl__(class_0_trades)
    class_2_pnl = __get_fees_free_pnl__(class_2_trades)

    return class_0_pnl, class_2_pnl

def plot_pnl_distribution(class_0_pnl, class_2_pnl):
    # Plot KDE for Class 0 and Class 2
    plt.figure(figsize=(10, 6))

    sns.kdeplot(class_0_pnl, label="Class 0 (Short Trades)", shade=True, color="red")
    sns.kdeplot(class_2_pnl, label="Class 2 (Long Trades)", shade=True, color="green")

    plt.title("PnL Distribution by Class (Kernel Density Estimate)")
    plt.xlabel("PnL")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def compute_log_returns(prices, prediction_class, rows_ahead=5):
    # Filter rows where the predictions are for the given class (0 or 2)
    class_rows = prices[prices['prediction'] == prediction_class]

    # Ensure we have enough rows to compute the future log return (5 rows ahead)
    class_rows['Mid_Future'] = class_rows['Mid'].shift(-rows_ahead)

    # Calculate log returns
    class_rows['Log_Return'] = np.log(class_rows['Mid_Future'] / class_rows['Mid'])

    # Drop rows with NaN values (where future price is not available)
    class_log_returns = class_rows['Log_Return'].dropna()

    return class_log_returns






def collect_log_returns(threshold_values, prices, probabilities):
    # Dictionary to store log-returns by class and threshold
    log_returns_by_class = {0: [], 2: []}

    # Loop through each threshold value
    for probability_threshold in threshold_values:
        # Prepare the prices data
        prices_copy = prices[['seconds', 'BIDp1', 'ASKp1', 'Mid']].copy()
        for i in range(probabilities.shape[1]):
            prices_copy[f'prob_{i}'] = probabilities[:, i]

        # Apply the threshold to make predictions
        prices_copy['prediction'] = prices_copy.apply(make_prediction, axis=1, threshold=probability_threshold)

        # Compute log-returns for class 0 and class 2 (5 rows ahead)
        class_0_log_returns = compute_log_returns(prices_copy, prediction_class=0, rows_ahead=5)
        class_2_log_returns = compute_log_returns(prices_copy, prediction_class=2, rows_ahead=5)

        # Store the log-returns
        log_returns_by_class[0].append(class_0_log_returns)
        log_returns_by_class[2].append(class_2_log_returns)

    return log_returns_by_class



def plot_log_returns_distributions(log_returns_by_class, threshold_values):

    returns_class_0 = np.concatenate([np.array(e) for e in log_returns_by_class[0]])
    mean_log_returns_0 = np.mean(returns_class_0)
    std_log_returns_0 = np.std(returns_class_0)

    returns_class_2 = np.concatenate([np.array(e) for e in log_returns_by_class[2]])
    mean_log_returns_2 = np.mean(returns_class_2)
    std_log_returns_2 = np.std(returns_class_2)

    x_min_0 = mean_log_returns_0 - 3 * std_log_returns_0
    x_max_0 = mean_log_returns_0 + 3 * std_log_returns_0

    x_min_2 = mean_log_returns_2 - 3 * std_log_returns_2
    x_max_2 = mean_log_returns_2 + 3 * std_log_returns_2

    # Set up the subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for the different thresholds
    colors = plt.cm.viridis(np.linspace(0, 1, len(threshold_values)))

    # Plot all log-returns for class 0 (left subplot)
    for idx, log_returns in enumerate(log_returns_by_class[0]):
        sns.kdeplot(log_returns, label=f"Threshold {threshold_values[idx]:.1f}", shade=False, color=colors[idx], ax=axes[0])

    # Plot all log-returns for class 2 (right subplot)
    for idx, log_returns in enumerate(log_returns_by_class[2]):
        sns.kdeplot(log_returns, label=f"Threshold {threshold_values[idx]:.1f}", shade=False, color=colors[idx], ax=axes[1])

    # Customize Class 0 subplot (left)
    axes[0].axvline(x=0, color='black', linestyle='--')
    axes[0].set_title("Class 0 (Down)")
    axes[0].set_xlabel("Log Returns")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(x_min_0, x_max_0)
    axes[0].grid(True)
    axes[0].legend(title="Probability Threshold")

    # Customize Class 2 subplot (right)
    axes[1].axvline(x=0, color='black', linestyle='--')
    axes[1].set_title("Class 2 (Up)")
    axes[1].set_xlabel("Log Returns")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(x_min_2, x_max_2)
    axes[1].grid(True)
    axes[1].legend(title="Probability Threshold")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



# Save the dictionary to a compressed pickle file
def save_compressed_pickle(data, file_path):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data successfully saved to {file_path}")



# Load the dictionary from a compressed pickle file
def load_compressed_pickle(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data




# Save the dictionary to a compressed pickle file
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data successfully saved to {file_path}")



# Load the dictionary from a compressed pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data




def convert_str_file_to_date(filename):
    dt_str1 = filename.split('.')[0][-8:]
    dt_str2 = dt_str1[:4] + '-' + dt_str1[4:6] + '-' + dt_str1[6:]
    return dt_str2


def get_data_path(symbol, H, T):
  # Define the directory where to extract the files
  init_path = '/content/drive/My Drive/LOB_Crypto/'
  extract_dir = init_path + 'data_compressed'
  extract_dir = os.path.join(extract_dir, symbol)

  # Data path
  dates_predH_path = [c.split('.zip')[0] for c in os.listdir(extract_dir) if (f'_H{H}_B{T}' in c)]
  try:
      data_path = os.path.join(extract_dir,dates_predH_path[0])
  except:
      print('No data found')

  return data_path


def get_evaluation_test_data_size(symbol, H, T):

    data_path = get_data_path(symbol, H, T)

    set_type = 'test'
    subset_ratio = 1

    X_list, y_list = [], []

    list_docs = [file_name for file_name in os.listdir(data_path) if (file_name.startswith(set_type) and file_name.endswith('.npz'))]
    total_length = len(list_docs)

    # Calculate the subset size (% of total data)
    subset_length = np.ceil(subset_ratio * total_length).astype(int)

    list_docs = random.sample(list_docs, subset_length)

    # Loop through each .npz file in the folder
    for file_name in sorted(list_docs):

        if file_name.startswith(set_type) and file_name.endswith('.npz'):

            file_path = os.path.join(data_path, file_name)

            # Load the .npz file
            data = np.load(file_path)

            y_list.append(data['y_data'])
            #print(file_name, sorted(os.listdir(data_path)).index(file_name), len(X_train_list))

            del data

    # Concatenate all the data from the lists into single numpy arrays
    y = np.concatenate(y_list, axis=0)

    del y_list
    import gc
    gc.collect()

    return y.shape


def adapt_models_list(models):
    model_aliases = {
        'binbtabl': 'binbtable',
        'binctabl': 'binctable'
    }
    return [model_aliases.get(model, model) for model in models]



def get_weighted_maker_taker_cost(key, trading_fees_dict):
    if 'market' in key:
        trader_typ = key.split("_")[1]
        return trading_fees_dict[trader_typ]
    elif "mixed" in key:
        # Extract proportions from the key
        try:
            proportion_taker = int(key.split("_")[1].replace("T", ""))
            taker_prop = proportion_taker / 100
            maker_prop = 1 - taker_prop
            # Compute the weighted trading fee
            #print(taker_prop, maker_prop)
            return taker_prop * trading_fees_dict['taker'] + maker_prop * trading_fees_dict['maker']
        except (IndexError, ValueError):
            raise ValueError(f"Invalid mixed key format: {key}")
    else:
        raise ValueError(f"Unknown key: {key}")


def compute_slippage(volatility, pAdv, k=0.05, alpha=1):
    """
    Computes the slippage per trade based on the given parameters.

    Parameters:
    - volatility (float): Standard deviation of returns.
    - pAdv (float): Proportion of average daily volume traded.
    - duration_trades (float): Duration of the trade in seconds.
    - k (float, optional): Impact coefficient (default: 0.05).
    - alpha (float, optional): Impact exponent for ADV (default: 1).

    Returns:
    - float: Slippage per trade.
    """
    # Compute slippage
    trade_cost_per_trade = k * volatility * (pAdv ** alpha)

    return trade_cost_per_trade



def get_trading_costs(df, trading_hyperparameters, trading_fees_dict):

    k = trading_hyperparameters['k']
    alpha = trading_hyperparameters['alpha']
    pAdv = trading_hyperparameters['pAdv']
    trading_type = trading_hyperparameters['mid_side_trading']

    volatility = df['ret'].std()
    med_tm, avg_tm = df['time_diff'].median(), df['time_diff'].mean()
    duration_per_timestep = med_tm if med_tm > 0 else avg_tm if avg_tm > 0 else 0.1
    volatility_per_day = volatility * np.sqrt(3600 * 24 / duration_per_timestep)

    duration_trades_per_days = df.groupby('date')['time_diff'].sum().median()
    nb_trades_per_day = df.groupby('date')['nb_trades'].sum().median()

    # spread cost
    coefficient = trading_hyperparameters['spread_coefficient']
    df['spreadCost'] = coefficient * df['spread_ret'] * df['nb_trades']

    # trading fee (log)
    trading_fee_per_trade = get_weighted_maker_taker_cost(trading_type, trading_fees_dict)
    df['feeCost'] = trading_fee_per_trade * df['nb_trades']
    df['feeCost'] = np.log(1 + df['feeCost'])  # spread_ret is already in log

    # slippage
    #slippage_taker_per_day = compute_slippage(volatility_per_day, pAdv, k=k, alpha=alpha)
    #slippage_taker_per_trade = slippage_per_day / nb_trades_per_day
    slippage_maker_per_trade = k * (df['spread_ret'] / 2) # Glosten-Milgrom model with Order Informational Weight = k
    slippage_taker_per_trade = 1.5 * slippage_maker_per_trade
    slippage_dict = {'maker': slippage_maker_per_trade, 'taker': slippage_taker_per_trade}
    slippage_per_trade = get_weighted_maker_taker_cost(trading_type, slippage_dict)
    df['slippage'] = slippage_per_trade * df['nb_trades']
    df['slippage'] = np.log(1 + df['slippage'])

    # sum of trading cost
    df['tradingCosts'] = df['spreadCost'] + df['feeCost'] + df['slippage']

    # compute net PnL
    df['PnL_net'] = df['PnL_gross'] - df['tradingCosts']

    return df



def compute_durations(trading_history_dataframe):
    duration_trades_arr = trading_history_dataframe['time_diff']
    duration_trades = duration_trades_arr.sum()
    duration_trades_year = duration_trades / (3600 * 24 * 365)
    duration_trades_day = duration_trades / (3600 * 24)
    return duration_trades, duration_trades_year, duration_trades_day, duration_trades_arr


def compute_trading_cost(trading_type, trading_fees_dict, volatility, pAdv, nb_trades, duration_trades, k, alpha):
    pAdv_per_trade = pAdv / nb_trades if nb_trades > 0 else 0
    trading_fee_per_trade, slippage_per_trade = get_trading_cost_per_trade(
        trading_type, trading_fees_dict, volatility, pAdv_per_trade, duration_trades, k, alpha
    )
    trading_cost_per_trade = trading_fee_per_trade + slippage_per_trade
    trading_cost_per_trade_log = np.log(1 - trading_cost_per_trade)
    return trading_cost_per_trade, trading_cost_per_trade_log, trading_fee_per_trade


def get_pnl_by_position(trading_history_dataframe):

    trading_history_dataframe['position_nb2'] = trading_history_dataframe['position_nb'].shift(1).bfill()

    log_gross_returns_by_position = trading_history_dataframe.groupby('position_nb2')['PnL_gross'].sum()
    trading_history_dataframe['position_gross_pnl'] = trading_history_dataframe['position_nb2'].map(log_gross_returns_by_position)
    trading_history_dataframe.loc[trading_history_dataframe['nb_trades'] < 2, 'position_gross_pnl'] = 0.0

    log_net_returns_by_position = trading_history_dataframe.groupby('position_nb2')['PnL_net'].sum()
    trading_history_dataframe['position_net_pnl'] = trading_history_dataframe['position_nb2'].map(log_net_returns_by_position)
    trading_history_dataframe.loc[trading_history_dataframe['nb_trades'] < 2, 'position_net_pnl'] = 0.0

    return trading_history_dataframe


def get_ror_by_position(trading_history_dataframe):

    trading_history_dataframe['position_gross_ror'] = np.exp(trading_history_dataframe['position_gross_pnl']) - 1
    trading_history_dataframe['position_net_ror'] = np.exp(trading_history_dataframe['position_net_pnl']) - 1

    return trading_history_dataframe


def get_yield_per_trade(trading_history_dataframe):
    trading_history_dataframe['position_nb2'] = trading_history_dataframe['position_nb'].shift(1).bfill()
    gross_yield_per_trade = (np.exp(trading_history_dataframe.groupby('position_nb2')['PnL_gross'].sum()) - 1).mean()
    net_yield_per_trade = (np.exp(trading_history_dataframe.groupby('position_nb2')['PnL_net'].sum()) - 1).mean()

    return gross_yield_per_trade, net_yield_per_trade


def compute_ror(total_log_pnl, nb_trades, trading_cost_per_trade_log, duration_trades_day, duration_trades_year):
    net_ror_day = np.exp(net_log_returns / duration_trades_day) - 1
    #net_ror_ann = np.exp(net_log_returns / duration_trades_year) - 1
    return net_log_returns, net_ror_day #, net_ror_ann



def get_profit_ratios(duration_adjust, risk_free_rate, returns):

    coef_time_adj = 365 * (3600 * 24 / duration_adjust)
    risk_free_rate_log_period = np.log(1 + risk_free_rate) / coef_time_adj

    sharpe_ratio_period, sortino_period = sharpe_ratio(returns, risk_free_rate_log_period), sortino_ratio(returns, risk_free_rate_log_period)
    sharpe_ratio_adj, sortino_adj = np.sqrt(coef_time_adj) * sharpe_ratio_period, np.sqrt(coef_time_adj) * sortino_period

    # Compute additional metrics
    max_drawdown = maximum_drawdown_geometric_returns(returns)
    win_loss = win_loss_ratio(returns)
    var = value_at_risk(returns)
    expected_shortfall_value = expected_shortfall(returns)

    dict_profit_measures = {
        'sharpe_ratio_period': sharpe_ratio_period,
        'sortino_ratio_period': sortino_period,
        'sharpe_ratio_adjusted': sharpe_ratio_adj,
        'sortino_ratio_adjusted': sortino_adj,
        'maximum_drawdown': max_drawdown,
        'win_loss_ratio': win_loss,
        'value_at_risk': var,
        'expected_shortfall': expected_shortfall_value
    }

    return dict_profit_measures



def compute_pnl_metrics(net_log_returns_arr, gross_yield_per_trade_bps, net_yield_per_trade_bps, net_ror_day, nb_trades_per_day , risk_free_rate_log_period, coef_adj_ratio):
    if len(net_log_returns_arr) > 0:
        return {
            'gross_yield_per_trade_bps': gross_yield_per_trade_bps,
            'net_yield_per_trade_bps': net_yield_per_trade_bps,
            'nb_trades_by_day': nb_trades_per_day,
            'net_ror_day': net_ror_day,
            'sharpe_ratio': coef_adj_ratio * sharpe_ratio(net_log_returns_arr, risk_free_rate=risk_free_rate_log_period),
            'sortino_ratio': coef_adj_ratio * sortino_ratio(net_log_returns_arr, risk_free_rate=risk_free_rate_log_period),
            'maximum_drawdown': maximum_drawdown_log_returns(net_log_returns_arr),
            'win_loss_ratio': win_loss_ratio(net_log_returns_arr),
            'value_at_risk': value_at_risk(net_log_returns_arr),
            'expected_shortfall': expected_shortfall(net_log_returns_arr)
        }
    else:
        print('trading_history_dataframe empty')
        return {
            'gross_yield_per_trade_bps':0.0,
            'nb_trades_by_day': 0,
            'cumulative_log_return': 0.0,
            'net_ror_day': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'maximum_drawdown': 0.0,
            'win_loss_ratio': 0.0,
            'value_at_risk': 0.0,
            'expected_shortfall': 0.0
        }


def compute_pnl_metrics2(condition, gross_yield_per_trade_bps, net_yield_per_trade_bps, nb_trades_per_day , average_period_position, dict_gross, dict_net):
    if condition:
        # Consolidate results
        metrics = {
            # Common metrics
            'gross_yield_per_trade_bps': gross_yield_per_trade_bps,
            'net_yield_per_trade_bps': net_yield_per_trade_bps,
            'nb_trades_by_day': nb_trades_per_day,
            'average_period_position': average_period_position,

            # Gross PnL Metrics
            **{f'gross_{key}': value for key, value in dict_gross.items()},

            # Net PnL Metrics
            **{f'net_{key}': value for key, value in dict_net.items()},
        }

        return metrics

    # Handle empty case
    else:
        print('trading_history_dataframe empty')
        return {
            'gross_yield_per_trade_bps': 0.0,
            'net_yield_per_trade_bps': 0.0,
            'nb_trades_by_day': 0,
            'average_period_position': 0.0,

            # Fill zeros for gross metrics
            **{f'gross_{key}': 0.0 for key in dict_gross.keys()},

            # Fill zeros for net metrics
            **{f'net_{key}': 0.0 for key in dict_net.keys()},
        }




def pnl_analysis(prices_model, trading_type, probability_threshold, spread_cost_capture_dict, trading_fees_dict, pAdv, k, alpha, risk_free_rate):
    trading_hyperparameters = {
        'mid_side_trading': trading_type,
        'probability_threshold': probability_threshold,
        'spread_coefficient': spread_cost_capture_dict[trading_type],
        'k': k,
        'alpha': alpha,
        'pAdv': pAdv
    }

    trading_history_dataframe = backtest2(prices_model, trading_hyperparameters)

    duration_trades, duration_trades_year, duration_trades_day, duration_trades_arr = compute_durations(trading_history_dataframe)

    nb_positions_history = trading_history_dataframe['nb_positions']
    nb_trades = 2*nb_positions_history.max() # 2 trades for each in-and-out position

    gross_log_pnl = trading_history_dataframe['PnL']
    total_log_gross_pnl = gross_log_pnl.sum()
    gross_yield_per_trade = np.exp(total_log_gross_pnl / nb_trades) - 1 if nb_trades > 0 else 0


    trading_cost_per_trade, trading_cost_per_trade_log, trading_fee_per_trade = compute_trading_cost(
        trading_history_dataframe, trading_fees_dict
    )

    #print(f'fee % of cost: {trading_fee_per_trade / trading_cost_per_trade:.6}')

    #net_log_returns, net_ror_day = compute_ror(
    #    total_log_gross_pnl, nb_trades, trading_cost_per_trade_log, duration_trades_day, duration_trades_year
    #)

    #print(f'{trading_type}: net RoR day: {net_ror_day:.10f}, net annualized RoR: {net_ror_ann:.6f}')

    net_log_returns_arr = gross_log_pnl + 2 * (nb_positions_history > 0) * trading_cost_per_trade_log
    total_log_net_pnl = net_log_returns_arr.sum()
    net_yield_per_trade = np.exp(total_log_net_pnl / nb_trades) - 1 if nb_trades > 0 else 0
    net_ror_day = np.exp(total_log_net_pnl / duration_trades_day) - 1

    risk_free_rate_log_period = np.log(1 + risk_free_rate) * duration_trades_year
    coef_adj_ratio = np.sqrt(1 / duration_trades_year)

    pnl_results = compute_pnl_metrics(net_log_returns_arr, gross_yield_per_trade, net_yield_per_trade, net_ror_day, nb_trades / duration_trades_day, risk_free_rate_log_period, coef_adj_ratio)

    if probability_threshold == 0.3:
        pnl_results.update({
            'returns_history': net_log_returns_arr,
            'duration_trade_history': duration_trades_arr
        })

    return pnl_results




def pnl_analysis2(prices_model, trading_type, probability_threshold, spread_cost_capture_dict, trading_fees_dict, pAdv, k, alpha, risk_free_rate):

        trading_hyperparameters = {
            'mid_side_trading': trading_type,
            'probability_threshold': probability_threshold,
            'spread_coefficient': spread_cost_capture_dict[trading_type],
            'k': k,
            'alpha': alpha,
            'pAdv': pAdv
        }


        trading_history_dataframe = backtest2(prices_model.copy(deep=True), trading_hyperparameters)

        condition_continue = trading_history_dataframe['nb_trades'].sum() > 1

        if condition_continue:

            # compute costs
            trading_history_dataframe = get_trading_costs(
                trading_history_dataframe, trading_hyperparameters, trading_fees_dict
            )

            # compute durations
            duration_trades, duration_trades_year, duration_trades_day, duration_trades_arr = compute_durations(trading_history_dataframe)

            # compute yield per trade
            trading_history_dataframe = get_pnl_by_position(trading_history_dataframe)
            trading_history_dataframe = get_ror_by_position(trading_history_dataframe)

            full_serie_ror_gross = np.exp(trading_history_dataframe['PnL_gross'])-1
            full_serie_ror_net = np.exp(trading_history_dataframe['PnL_net'])-1

            positions_df = trading_history_dataframe.loc[trading_history_dataframe['nb_trades'] > 1]
            gross_yield_per_trade_bps = 1e4 * positions_df['position_gross_ror'].mean() / 2
            net_yield_per_trade_bps = 1e4 * positions_df['position_net_ror'].mean() / 2

            med_time_per_position = trading_history_dataframe.groupby('position_nb2')['time_diff'].sum().median()
            nb_trades_per_day = trading_history_dataframe['nb_trades'].sum() / duration_trades_day

            # sharpe and sortino ratios
            med_tm, avg_tm = trading_history_dataframe['time_diff'].median(), trading_history_dataframe['time_diff'].mean()
            duration_adjust = med_tm if med_tm > 0 else avg_tm if avg_tm > 0 else 0.1
            dict_profit_measures_gross = get_profit_ratios(duration_adjust, risk_free_rate, trading_history_dataframe['PnL_gross'].to_numpy())
            dict_profit_measures_net = get_profit_ratios(duration_adjust, risk_free_rate, trading_history_dataframe['PnL_net'].to_numpy())

        else:
            print('0 closed position')
            full_serie_ror_gross = np.array([])
            full_serie_ror_net = np.array([])
            gross_yield_per_trade_bps = 0.0
            net_yield_per_trade_bps = 0.0
            nb_trades_per_day = 0
            med_time_per_position = 0.0
            dict_keys_list = get_profit_ratios(1, 0.02, np.array([1.0,2.0])).keys()
            dict_profit_measures_gross = {key: 0.0 for key in dict_keys_list}
            dict_profit_measures_net = {key: 0.0 for key in dict_keys_list}

        #pnl_results = compute_pnl_metrics(net_log_returns_arr, gross_yield_per_trade, net_yield_per_trade, net_ror_day, nb_trades / duration_trades_day, risk_free_rate_log_period, coef_adj_ratio)
        pnl_results = compute_pnl_metrics2(
                                           condition_continue,
                                           gross_yield_per_trade_bps,
                                           net_yield_per_trade_bps,
                                           nb_trades_per_day,
                                           med_time_per_position,
                                           dict_profit_measures_gross,
                                           dict_profit_measures_net
                                           )

        if probability_threshold == 0.3:
            pnl_results.update({
                    'net_pnl_history': np.exp(trading_history_dataframe['PnL_net'])-1,
                })
            if trading_type == 'market_taker':
                pnl_results.update({
                    'gross_pnl_history': np.exp(trading_history_dataframe['PnL_gross'])-1
                })


        return pnl_results



def run_backtest(config_path):

    with open(config_path, "r") as f:
        config = json.load(f)

    table = config.get("table", "book")
    seed = config.get("seed", 42)
    symbols = config["symbols"]
    models = adapt_models_list(config["models"])
    exchanges = config["exchanges"]
    path_sqlite = config["path_sqlite"]
    path_output_dir = os.path.join(os.getcwd(), config["path_save_results"])
    root_path = os.path.join(os.getcwd(), config["path_save_dataset"])
    set_type = 'test'
    spread_cost_capture_dict = config["trader_types"]
    trader_types = list(spread_cost_capture_dict.keys())
    probability_thresholds = config["probability_thresholds"]
    type_library = config.get("type_library", "pt")
    risk_free_rate = config.get("risk_free_rate", 0.02)
    trading_fees_dict = config.get("trading_fee", None)
    pAdv = config.get("pAdv", 1e-4)
    k = config.get("k", 0.05)
    alpha = config.get("alpha", 1)

    np.random.seed(seed)

    for symbol in symbols:
        print(f"Symbol: {symbol}")
        symbol_path = os.path.join(root_path, symbol)
        filenames_tm_horizon = sorted(
            [folder for folder in os.listdir(symbol_path) if os.path.isdir(os.path.join(symbol_path, folder))]
        )

        for filename in filenames_tm_horizon:
            parts_file = filename.split('_')
            tm_horizon = f'{parts_file[-2]}_{parts_file[-1]}'
            print(f"Time Horizon: {tm_horizon}")

            root = os.path.join(symbol_path, filename)
            seq_file_path = os.path.join(root, 'all_sequence_numbers.pkl')

            if not os.path.exists(seq_file_path):
                continue

            with open(seq_file_path, 'rb') as f:
                all_sequence_numbers = pickle.load(f)

            test_dates = [
                convert_str_file_to_date(file_name) for file_name in os.listdir(root)
                if file_name.startswith(set_type) and file_name.endswith('.npz')
            ]
            test_dates = np.intersect1d(all_sequence_numbers.columns.tolist(), test_dates).tolist()
            backtest_data = get_backtest_data(table, test_dates, [symbol], exchanges, path_sqlite)

            prices = pd.concat([
                backtest_data.loc[
                    backtest_data['sequence_number'].isin(all_sequence_numbers[date].dropna())
                ] for date in test_dates
            ]).reset_index(drop=True).sort_values('sequence_number').copy(deep=True)

            columns_keep = ['date', 'origin_dt', 'sequence_number', 'Mid', 'BIDp1', 'ASKp1','bid_0_size','ask_0_size']
            prices = prices[columns_keep]
            prices['spread'] = prices['ASKp1'] - prices['BIDp1']
            prices['spread_ret'] = np.log(prices['ASKp1'] / prices['BIDp1'])
            prices['ret'] = np.log(prices['Mid'] / prices['Mid'].shift(1))
            prices['time_diff'] = prices['origin_dt'].diff().dt.total_seconds().fillna(0.0)

            for model in models:
                print(f"Model: {model}")
                model_folder_path = os.path.join(root, model)

                if not os.path.exists(model_folder_path):
                    print(f"Model folder does not exist: {model_folder_path}")
                    continue

                checkpoint_path = (
                    get_latest_checkpoint_old(model, model_folder_path, type_library, seed)
                    if 'binbtable' in models or 'binctable' in models else
                    get_latest_checkpoint(model, model_folder_path, type_library, seed)
                )

                directory, file_name = os.path.split(checkpoint_path)
                new_file_name = file_name.replace(model, 'test_results').replace('.ckpt', '.pkl').replace('seed=42-', 'seed=42_')
                new_path = os.path.join(directory, new_file_name)

                with open(new_path, 'rb') as f:
                    test_results = pickle.load(f)

                if test_results is None:
                    print(f"No test results found in {model_folder_path}")
                    continue

                prices_model = prices.copy(deep=True)
                probabilities = np.array(test_results['probabilities'])

                for i in range(probabilities.shape[1]):
                    prices_model[f'prob_{i}'] = probabilities[:, i]

                model_results = {}

                for probability_threshold in probability_thresholds:
                    threshold_results = {}

                    prices_model['prediction'] = prices_model.apply(
                        make_prediction, axis=1, threshold=probability_threshold
                    )

                    for trading_type in trader_types:
                        print(f"Trading type: {trading_type}, Threshold: {probability_threshold}")

                        #pnl_results = pnl_analysis(prices_model, trading_type, probability_threshold, spread_cost_capture_dict, trading_fees_dict, pAdv, k, alpha, risk_free_rate)
                        pnl_results = pnl_analysis2(prices_model, trading_type, probability_threshold, spread_cost_capture_dict, trading_fees_dict, pAdv, k, alpha, risk_free_rate)
                        print(pnl_results['net_yield_per_trade_bps'])

                        threshold_results[trading_type] = pnl_results
                        del pnl_results
                        gc.collect()

                    model_results[probability_threshold] = threshold_results

                path_save_dict = os.path.join(path_output_dir, f'backtest_{symbol}_{tm_horizon}_{model}.pkl.gz')
                save_compressed_pickle(model_results, path_save_dict)

                del test_results, probabilities, prices_model, model_results
                gc.collect()


            del prices, all_sequence_numbers
            gc.collect()



def aggregate_backtest_data(config_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract configuration parameters
    root_path = os.path.join(os.getcwd(), config["path_save_dataset"])
    path_output_dir = config["path_save_results"]
    symbols = config["symbols"]
    models = adapt_models_list(config["models"])
    trader_types = list(config["trader_types"].keys())
    metrics = [
        'gross_yield_per_trade_bps', 'net_yield_per_trade_bps', 'nb_trades_by_day',
        'average_period_position', 'gross_sharpe_ratio_period', 'gross_sortino_ratio_period',
        'gross_sharpe_ratio_adjusted', 'gross_sortino_ratio_adjusted', 'gross_maximum_drawdown',
        'gross_win_loss_ratio', 'gross_value_at_risk', 'gross_expected_shortfall',
        'net_sharpe_ratio_period', 'net_sortino_ratio_period', 'net_sharpe_ratio_adjusted',
        'net_sortino_ratio_adjusted', 'net_maximum_drawdown', 'net_win_loss_ratio',
        'net_value_at_risk', 'net_expected_shortfall'
    ]

    # Initialize data storage
    aggregated_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    # Process symbols
    for symbol in symbols:
        print(f"Processing Symbol: {symbol}")
        symbol_path = os.path.join(root_path, symbol)
        filenames_tm_horizon = sorted(
            [folder for folder in os.listdir(symbol_path) if os.path.isdir(os.path.join(symbol_path, folder))]
        )

        # Process time horizons
        for filename in filenames_tm_horizon:
            parts_file = filename.split('_')
            tm = parts_file[-2]
            tm_horizon = f'{parts_file[-2]}_{parts_file[-1]}'
            print(f"Processing Time Horizon: {tm}")

            # Process models
            for model in models:
                model_aliases = {'binbtabl': 'binbtable', 'binctabl': 'binctable'}
                model_resolved = model_aliases.get(model, model)
                path_save_dict = os.path.join(path_output_dir, f'backtest_{symbol}_{tm_horizon}_{model_resolved}.pkl.gz')

                if not os.path.exists(path_save_dict):
                    print(f"File not found: {path_save_dict}")
                    continue

                # Load results
                PnL_results = load_compressed_pickle(path_save_dict)
                probability_thresholds = list(PnL_results.keys())

                # Extract metrics
                for trader_type in trader_types:
                    if trader_type not in PnL_results[0.3]:
                        continue

                    for metric in metrics:
                        if metric in PnL_results[0.3][trader_type]:
                            aggregated_data[model][tm][trader_type][symbol][metric] = [
                                PnL_results[threshold][trader_type].get(metric, None)
                                for threshold in probability_thresholds
                            ]

                    # Handle special cases for pnl history
                    if 'net_pnl_history' in PnL_results[0.3][trader_type]:
                        aggregated_data[model][tm][trader_type][symbol]['net_pnl_history'] = [
                            PnL_results[threshold][trader_type]['net_pnl_history']
                            for threshold in probability_thresholds if 'net_pnl_history' in PnL_results[threshold][trader_type]
                        ]
                    if trader_type == 'market_taker' and 'gross_pnl_history' in PnL_results[0.3][trader_type]:
                        aggregated_data[model][tm][trader_type][symbol]['gross_pnl_history'] = [
                            PnL_results[threshold][trader_type]['gross_pnl_history']
                            for threshold in probability_thresholds if 'gross_pnl_history' in PnL_results[threshold][trader_type]
                        ]

    return aggregated_data



def plot_model_metrics(config_path, aggregated_data):

    # Load configuration file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract relevant parameters from the configuration
    models = adapt_models_list(config["models"])
    trader_types_init = list(config["trader_types"].keys())
    time_horizons = ['H'+str(h) for h in config["pred_horizons"]]
    symbols = config["symbols"]
    colors = config["plot_colors"]
    symbolics = config["plot_symbolics"]
    probability_thresholds = config["probability_thresholds"]
    path_plot_dir = config["path_save_plots"]
    # Metrics to be plotted
    metrics = config["metrics_pnl"]
    labels = config["metrics_pnl_labels"]
    metric_labels = dict(zip(metrics, labels))


    # Loop plot
    trader_types = ['no_cost'] + trader_types_init
    for model in models:
        for trader_type_init in trader_types:
            if trader_type_init == 'no_cost':
                trader_type = 'market_taker'
                pnl_type = 'gross'
            else:
                trader_type = trader_type_init
                pnl_type = 'net'

            specific_metrics = ['nb_trades_by_day', 'average_period_position'] + [c for c in metrics if c.split('_')[0]  == pnl_type]

            # Compute yield_per_day_bps if missing
            for symbol in symbols:
                for time_horizon in time_horizons:
                    yield_by_day_label = f'{pnl_type}_yield_per_day_pct'
                    if yield_by_day_label not in aggregated_data[model][time_horizon][trader_type][symbol]:
                        trades = aggregated_data[model][time_horizon][trader_type][symbol].get(f'{pnl_type}_yield_per_trade_bps', np.nan)
                        nb_trades = aggregated_data[model][time_horizon][trader_type][symbol].get('nb_trades_by_day', np.nan)
                        aggregated_data[model][time_horizon][trader_type][symbol][yield_by_day_label] = 1e-2 * np.array(trades) * np.array(nb_trades)

            fig, axes = plt.subplots(len(specific_metrics), len(time_horizons), figsize=(10, 15), sharex=True)

            # Adjusting the title and subtitle for the side type
            fig.suptitle(f'{model.upper()}', fontsize=12, fontweight='bold', y=0.945)
            trading_title = trader_type_init.replace('_', ' ')
            fig.text(0.5, 0.923, f'{trading_title.capitalize()}', ha='center', fontsize=10, fontweight='normal', style='italic')

            for metric_idx, metric in enumerate(specific_metrics):
                for horizon_idx, time_horizon in enumerate(time_horizons):

                    ax = axes[metric_idx, horizon_idx]

                    tabl = np.array([
                        aggregated_data[model][time_horizon][trader_type][symbol].get(metric, np.nan)
                        for symbol in symbols
                    ])

                    mean_tabl = np.nanmean(tabl, axis=0)
                    std_tabl = np.nanstd(tabl, axis=0)

                    y_label = metric_labels.get(metric, metric).replace('Net ','').replace('Gross ','').replace('pct','%')

                    # Set logarithmic scale for specific metrics
                    if metric  == 'average_period_position': #,f'{pnl_type}_maximum_drawdown']:
                        #ax.set_yscale('symlog', linthresh=0.01)
                        ax.set_yscale('log')
                        y_label += ' (log)'
                    elif metric == f'{pnl_type}_maximum_drawdown':
                        ax.set_yscale('log')
                        ax.invert_yaxis()  # Flip the axis to display negatives
                        tabl = -tabl  # Flip values for plotting
                        mean_tabl = -mean_tabl
                        std_tabl = -std_tabl
                        y_label += ' (log)'
                        ax.yaxis.set_major_formatter(mticker.FuncFormatter(neg_log_formatter))

                    for i, (symbol, symbol_color, symbol_marker) in enumerate(zip(symbols, colors, symbolics)):
                        if metric in aggregated_data[model][time_horizon][trader_type][symbol]:
                            ax.plot(probability_thresholds, tabl[i], linestyle='None', marker=symbol_marker, color=symbol_color, label=symbol.split('-')[0])

                    avg_plot_exceptions = [f'{pnl_type}_maximum_drawdown', f'{pnl_type}_value_at_risk', f'{pnl_type}_expected_shortfall']
                    if metric not in avg_plot_exceptions:
                        ax.plot(probability_thresholds, mean_tabl, color='grey', linewidth=1.5, linestyle='--', label='Mean')
                        ax.fill_between(probability_thresholds, mean_tabl - std_tabl, mean_tabl + std_tabl, color='gray', alpha=0.1)


                    if horizon_idx == 0:
                        y_label_split = y_label.split(' ')
                        y_label_lines = y_label.replace(' ', '\n', 1) if len(y_label_split) < 3 else ' '.join(y_label_split[:int(len(y_label_split)/2)]) + '\n' + ' '.join(y_label_split[int(len(y_label_split)/2):])
                        ax.set_ylabel(y_label_lines, fontsize=8, fontweight='bold', labelpad=10)
                    if metric_idx == len(specific_metrics) - 1:
                        ax.set_xlabel('Probability Threshold', fontsize=10, fontweight='bold')
                    if metric_idx == 0:
                        ax.set_title(time_horizon, fontsize=10, fontweight='bold')

            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles[:len(symbols)], labels[:len(symbols)], loc='upper center', ncol=len(symbols), fontsize=9, bbox_to_anchor=(0.81, 0.94), handletextpad=0.15, columnspacing=0.5)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.subplots_adjust(wspace=0.40, hspace=0.15)
            plt.savefig(os.path.join(path_plot_dir, f'metrics_pnl_vs_proba_threshold_{model}_{trader_type_init}.png'))
            plt.show()
            #plt.close(fig)



def plot_trader_types_view(config_path, aggregated_data):

    # Load configuration file
    with open(config_path, "r") as f:
        config = json.load(f)

    models = adapt_models_list(config["models"])
    trader_types_init = list(config["trader_types"].keys())
    time_horizons = ['H'+str(h) for h in config["pred_horizons"]]
    symbols = config["symbols"]
    colors = config["plot_colors"]
    symbolics = config["plot_symbolics"]
    path_plot_dir = config["path_save_plots"]

    trader_type_labels = {
        'market_taker': 'Taker',
        'market_maker': 'Maker',
        'mixed_25T_75M': '25%T',
        'mixed_50T_50M': '50%T',
        'mixed_75T_25M': '75%T'
    }

    metric = "net_yield_per_trade_bps"

    fig, axes = plt.subplots(len(symbols), len(time_horizons), figsize=(7, 10), sharex=True)
    fig.suptitle(f'Net Yield per Trade (bps)', fontsize=12, fontweight='bold', y=0.98)
    fig.text(0.5, 0.95, 'slippage: (maker: 0.25, taker: 0.375) spread | exch fees: (maker: -0.2, taker: 2.1) bps | spread capture: (maker: -1, taker: 1) spread',
             ha='center', fontsize=6, style='italic', color='dimgray')

    for symbol_idx, symbol in enumerate(symbols):
        avg_gross_values = {th: np.mean([aggregated_data[model][th]['market_taker'][symbol]['gross_yield_per_trade_bps'][0] for model in models if 'gross_yield_per_trade_bps' in aggregated_data[model][th]['market_taker'][symbol]]) for th in time_horizons}
        for horizon_idx, time_horizon in enumerate(time_horizons):
            avg_gross_value = avg_gross_values[time_horizon]
            ax = axes[symbol_idx, horizon_idx]
            x_labels = []
            y_values = []
            model_markers = []
            model_colors = []
            model_labels = []

            for trader_type in ['market_taker', 'mixed_75T_25M', 'mixed_50T_50M', 'mixed_25T_75M', 'market_maker']:
                for model, symbol_marker, symbol_color in zip(models, symbolics, colors):
                    if metric in aggregated_data[model][time_horizon][trader_type][symbol]:
                        value = aggregated_data[model][time_horizon][trader_type][symbol][metric][0]
                        y_values.append(value)
                        x_labels.append(trader_type_labels[trader_type])
                        model_markers.append(symbol_marker)
                        model_colors.append(symbol_color)
                        model_labels.append(model)

            for x, y, marker, color, label in zip(x_labels, y_values, model_markers, model_colors, model_labels):
                ax.plot(x, y, linestyle='None', marker=marker, color=color, label=label)
            ax.axhline(avg_gross_value, color='gray', linestyle='--', linewidth=1, label='Gross Avg')

            if horizon_idx == 0:
                ax.set_ylabel(symbol.split('-')[0].upper(), fontsize=10, fontweight='bold')
            if symbol_idx == len(symbols) - 1:
                ax.set_xticks(range(len(trader_type_labels)))
                ax.set_xticklabels(
                    [trader_type_labels[t].replace(' ', '') for t in ['market_taker', 'mixed_75T_25M', 'mixed_50T_50M', 'mixed_25T_75M', 'market_maker']],
                    fontsize=8, fontweight='bold'
                )
            if symbol_idx == 0:
                ax.set_title(time_horizon, fontsize=10, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:len(models)], labels[:len(models)], loc='upper center', ncol=len(models), fontsize=9, bbox_to_anchor=(0.7, 0.94), handletextpad=0.1, columnspacing=0.4)
    fig.legend([plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1)], ['Gross yield avg across models'], loc='upper center', fontsize=9, bbox_to_anchor=(0.23, 0.94), handletextpad=0.15, columnspacing=0.5, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(wspace=0.25, hspace=0.1)
    plt.savefig(os.path.join(path_plot_dir, 'yield_per_trade.png'))
    plt.show()
    #plt.close(fig)



def plot_gross_metrics_by_pred_horizon(config_path, aggregated_data):
    # Load configuration file
    with open(config_path, "r") as f:
        config = json.load(f)

    models = adapt_models_list(config["models"])
    time_horizons = ['H' + str(h) for h in config["pred_horizons"]]
    symbols = config["symbols"]
    colors = config["plot_colors"]
    symbolics = config["plot_symbolics"]
    path_plot_dir = config["path_save_plots"]

    metrics = [
        "gross_yield_per_day_bps",
        "gross_sharpe_ratio_adjusted",
        "gross_sortino_ratio_adjusted",
        "gross_value_at_risk",
        "gross_maximum_drawdown"
    ]

    labels = [
        "Yield per Day (bps)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Value at Risk (5%)",
        "Maximum Drawdown"
    ]
    metric_labels = dict(zip(metrics, labels))

    fig, axes = plt.subplots(len(symbols), len(metrics), figsize=(10,10), sharex=True)
    fig.suptitle('Across Trading Horizons - No trading Costs', fontsize=14, fontweight='bold', y = 0.95)
    fig.text(0.5, 0.91, f'No Trading Costs', ha='center', fontsize=10, fontweight='normal', style='italic')

    for model in models:
        for symbol in symbols:
            for time_horizon in time_horizons:
                yield_by_day_label = 'gross_yield_per_day_bps'
                if yield_by_day_label not in aggregated_data[model][time_horizon]['market_taker'][symbol]:
                    trades = aggregated_data[model][time_horizon]['market_taker'][symbol].get('gross_yield_per_trade_bps', np.nan)
                    nb_trades = aggregated_data[model][time_horizon]['market_taker'][symbol].get('nb_trades_by_day', np.nan)
                    aggregated_data[model][time_horizon]['market_taker'][symbol][yield_by_day_label] = np.array(trades) * np.array(nb_trades)

    for symbol_idx, symbol in enumerate(symbols):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[symbol_idx, metric_idx]

            x_labels = []
            y_values = []
            model_markers = []
            model_colors = []
            model_labels = []

            for h_idx, time_horizon in enumerate(time_horizons):
                for model, symbol_marker, symbol_color in zip(models, symbolics, colors):
                    if metric in aggregated_data[model][time_horizon]['market_taker'][symbol]:
                        value = aggregated_data[model][time_horizon]['market_taker'][symbol][metric][0]
                        y_values.append(value)
                        x_labels.append(time_horizon)
                        model_markers.append(symbol_marker)
                        model_colors.append(symbol_color)
                        model_labels.append(model)


            y_label = metric_labels[metric]
            if metric in ["gross_maximum_drawdown"]: #, "gross_value_at_risk"]:
                ax.set_yscale("log")
                ax.invert_yaxis()
                y_values = [-y for y in y_values]
                y_label += ' (log)'
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(neg_log_formatter))

            if metric in ["gross_value_at_risk"]:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(scientific_formatter))

            for x, y, marker, color, label in zip(x_labels, y_values, model_markers, model_colors, model_labels):
                ax.plot(x, y, linestyle='None', marker=marker, color=color, label=label)


            if symbol_idx == len(symbols) - 1:
                ax.set_xlabel("Time Horizon", fontsize=8, fontweight='bold')
            if metric_idx == 0:
                ax.set_ylabel(symbol.split('-')[0].upper(), fontsize=8, fontweight='bold')
            if symbol_idx == 0:
                ax.set_title(metric_labels[metric], fontsize=10, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[:len(models)], labels[:len(models)], loc='upper center', ncol=len(models), fontsize=9, 
        bbox_to_anchor=(0.78, 0.93), handletextpad=0.15, columnspacing=0.5
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(wspace=0.6, hspace=0.2)
    plt.savefig(os.path.join(path_plot_dir, 'across_prediction_horizons_gross.png'))
    plt.show()
    #plt.close(fig)



def plot_gross_pnl_history(config_path, aggregated_data):
    # Load configuration file
    with open(config_path, "r") as f:
        config = json.load(f)

    models = adapt_models_list(config["models"])
    time_horizons = ['H' + str(h) for h in config["pred_horizons"]]
    symbols = config["symbols"]
    colors = config["plot_colors"]
    path_plot_dir = config["path_save_plots"]

    fig, axes = plt.subplots(len(symbols), len(time_horizons), figsize=(7, 10))
    fig.suptitle('Cumulative PnL', fontsize=14, fontweight='bold', y = 0.97)
    fig.text(0.5, 0.935, f'No Trading Costs', ha='center', fontsize=10, fontweight='normal', style='italic')

    for symbol_idx, symbol in enumerate(symbols):
        for horizon_idx, time_horizon in enumerate(time_horizons):
            ax = axes[symbol_idx, horizon_idx]
            max_x = 0

            for model, color in zip(models, colors):
                if 'gross_pnl_history' in aggregated_data[model][time_horizon]['market_taker'][symbol]:
                    pnl_history = aggregated_data[model][time_horizon]['market_taker'][symbol]['gross_pnl_history'][0]
                    cum_pnl_history = np.cumsum(pnl_history)
                    x_series = np.arange(len(cum_pnl_history))
                    max_x = max(max_x, len(cum_pnl_history))
                    ax.plot(x_series, cum_pnl_history, linestyle='-', color=color, label=model)

            format_xticks(ax, max_x)

            if horizon_idx == 0:
                ax.set_ylabel(symbol.split('-')[0].upper(), fontsize=8, fontweight='bold')
            if symbol_idx == len(symbols) - 1:
                ax.set_xlabel('Time Steps', fontsize=8, fontweight='bold')
            if symbol_idx == 0:
                ax.set_title(time_horizon, fontsize=10, fontweight='bold')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[:len(models)], labels[:len(models)], loc='upper center', ncol=len(models), fontsize=9, 
        bbox_to_anchor=(0.70, 0.93), handletextpad=0.15, columnspacing=0.5
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(wspace=0.3, hspace=0.2)
    plt.savefig(os.path.join(path_plot_dir, 'cum_pnl_history_gross.png'))
    plt.show()
    #plt.close(fig)



def plot_results_backtests(config_path, aggregated_data):
    plot_model_metrics(config_path, aggregated_data)
    plot_trader_types_view(config_path, aggregated_data)
    plot_gross_metrics_by_pred_horizon(config_path, aggregated_data)
    plot_gross_pnl_history(config_path, aggregated_data)


def run_pipeline_backtest(config_path, run_backtest = True):

    # Compute Returns and Costs historically
    if run_backtest:
        run_backtest(config_path)

    # Aggregate data
    aggregated_data = aggregate_backtest_data(config_path)
    
    # Plot results
    plot_results_backtests(config_path, aggregated_data)

