import os
import sys
import json
import gc
import pickle
import random
import sqlite3
import lz4
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

from utils2 import *


def get_dates(table, symbols, exchanges, start_date, end_date, path_sqlite, min_rows=5000):
    """
    Fetch unique dates with at least `min_rows` rows where 'bid_0_price' is not null between start_date and end_date.

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
    db_name = os.path.join(path_sqlite, f"{table}.sqlite")
    conn = sqlite3.connect(db_name)
    valid_dates = []

    for symbol in symbols:
        for exchange in exchanges:
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
            result = pd.read_sql_query(query, conn, params=(start_date, end_date, symbol, exchange, min_rows))
            if not result.empty:
                valid_dates += result['date'].tolist()

    conn.close()
    return sorted(valid_dates)


def split_dates(dates: List[str], train_ratio=0.6, val_ratio=0.5) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of dates into train, validation, and test sets.

    Parameters:
    - dates (list): A list of available dates in sorted order.
    - train_ratio (float): Proportion of data for training (default is 0.6).
    - val_ratio (float): Proportion of the remaining data after training for validation (default is 0.2).

    Returns:
    - train_dates, val_dates, test_dates: Lists of dates for each split, ensuring both validation and test sets
      have at least one date.

    Adjusts the splits if the validation or test set would have fewer than one date.
    """
    # Ensure dates are sorted
    dates = sorted(dates)
    n_total_dates = len(dates)

    # Calculate split indices
    n_train = int(n_total_dates * train_ratio)
    n_val = int((n_total_dates - n_train) * val_ratio)
    n_test = n_total_dates - n_train - n_val

    # Split the dates into train, validation, and test sets
    train_dates = dates[:n_train]
    val_dates = dates[n_train:n_train + n_val]
    test_dates = dates[n_train + n_val:]

    if len(val_dates) < 1 or len(test_dates) < 1:
        raise ValueError(
            f"Validation and test sets must each have at least one date.\n"
            f"Current split: {len(train_dates)} train, {len(val_dates)} validation, {len(test_dates)} test.\n"
            f"Total dates: {n_total_dates}. Please adjust train_ratio or val_ratio."
        )

    # Re-sort validation and test sets to preserve date order
    val_dates = sorted(val_dates)
    test_dates = sorted(test_dates)

    return train_dates, val_dates, test_dates




def rearrange_dates(dic_dates, val_date_st, test_date_st):
    """
    Rearrange training, validation, and testing dates to match specific starting points.
    """
    val_date_st_dt = pd.to_datetime(val_date_st)
    test_date_st_dt = pd.to_datetime(test_date_st)

    all_dates = pd.to_datetime(dic_dates['train_dates'] + dic_dates['val_dates'] + dic_dates['test_dates'])
    train_dates = all_dates[all_dates < val_date_st_dt].strftime('%Y-%m-%d').tolist()
    val_dates = all_dates[(all_dates >= val_date_st_dt) & (all_dates < test_date_st_dt)].strftime('%Y-%m-%d').tolist()
    test_dates = all_dates[all_dates >= test_date_st_dt].strftime('%Y-%m-%d').tolist()

    return train_dates, val_dates, test_dates, {'train_dates': train_dates, 'val_dates': val_dates, 'test_dates': test_dates}


def fetch_data_between_dates(table, symbols, exchanges, start_date, end_date, path_sqlite):
    """
    Fetch data for specified symbols and exchanges between given dates from a SQLite database.
    """
    base_columns = ['date', 'origin_time', 'received_time', 'sequence_number', 'symbol', 'exchange']
    db_name = os.path.join(path_sqlite, f"{table}.sqlite")
    conn = sqlite3.connect(db_name)
    all_data = []

    for symbol in symbols:
        for exchange in exchanges:
            query = f"""
            SELECT *
            FROM {table}
            WHERE date BETWEEN ? AND ?
              AND symbol = ?
              AND exchange = ?
            """
            result = pd.read_sql_query(query, conn, params=(start_date, end_date, symbol, exchange))
            if not result.empty:
                all_data.append(result)

    conn.close()
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def rearrange_features_names(data):

    old_features_names = data.columns[data.columns.str.contains('bid_') + data.columns.str.contains('ask_')]
    other_cols = [c for c in data.columns if c not in old_features_names]

    ASKp = data.columns[data.columns.str.contains(r'(?=.*ask)(?=.*price)')].tolist()
    ASKs = data.columns[data.columns.str.contains(r'(?=.*ask)(?=.*size)')].tolist()
    BIDp = data.columns[data.columns.str.contains(r'(?=.*bid)(?=.*price)')].tolist()
    BIDs = data.columns[data.columns.str.contains(r'(?=.*bid)(?=.*size)')].tolist()

    data = data[other_cols + ASKp + ASKs + BIDp + BIDs]

    feature_names = []
    levels = int(len(old_features_names) / 4)
    feature_names_raw = [
                "ASKp",
                "ASKs",
                "BIDp",
                "BIDs",
            ]  # Define sorted raw features' names.

    feature_names = []
    for j in range(4):
        for i in range(0, levels):
            feature_names += [
                feature_names_raw[j] + str(i)
            ]  # Add to raw features' names the level number.

    # Update the column names of your dataframe
    new_cols = other_cols + feature_names
    data.columns = new_cols

    feature_names2 = []
    for i in range(0, levels):
        for j in range(4):
            feature_names2 += [
                feature_names_raw[j] + str(i)
            ]  # Add to raw features' names the level number.


    new_cols2 = other_cols + feature_names2
    data = data[new_cols2]

    return data, feature_names


def get_log_volume(data):
    """
    Convert volume features to log scale.
    """
    volume_cols = data.columns[data.columns.str.contains('BIDs') | data.columns.str.contains('ASKs')]
    data[volume_cols] = np.log1p(data[volume_cols])
    return data


def get_mid_px(data):
    """
    Compute the mid-price as the average of the best bid and ask prices.
    """
    data['mid_px'] = (data['ASKp0'] + data['BIDp0']) / 2
    return data


def compute_price_diff_target(data, h, type_target, tick_size, type_mean):
    """
    Compute the target variable (price difference) for a given prediction horizon `h`.
    """
    data[f"Raw_Target_{h}"] = data['mid_px'].shift(-h) - data['mid_px']
    if type_mean == 'simple':
        data[f"Smooth_Target_{h}"] = data['mid_px'].rolling(window=h).mean().shift(-h) - data['mid_px'].rolling(window=h).mean()
    elif type_mean == 'ewa':
        data[f"Smooth_Target_{h}"] = data['mid_px'].ewm(span=h).mean().shift(-h) - data['mid_px'].ewm(span=h).mean()

    data['y'] = np.where(data[f"{type_target}_Target_{h}"] > tick_size, 2,
                         np.where(data[f"{type_target}_Target_{h}"] < -tick_size, 0, 1))
    return data



def extend_bins(flat_series_ref, bins_type, z_extensions):

    # Calculate mean, std, and max/min of the prior distribution
    mean = np.mean(flat_series_ref)
    std = np.std(flat_series_ref)
    max_value = np.max(flat_series_ref)
    min_value = np.min(flat_series_ref)

    # Compute z_max and z_min (z-scores of max and min values)
    z_max = (max_value - mean) / std
    z_min = (min_value - mean) / std

    # Create extended bins on the right side (above max)
    extended_bins_right = [(z_max + z_ext) * std + mean for z_ext in z_extensions]

    # Create extended bins on the left side (below min)
    extended_bins_left = [(z_min - z_ext) * std + mean for z_ext in z_extensions]

    # Combine bins: include the original bin edges (pd.cut will handle this automatically)
    all_bins = np.concatenate((extended_bins_left[::-1], bins_type, extended_bins_right))

    return all_bins



def apply_binning_cut(feature_values, new_bins, n_bins):

    cut_results = pd.cut(feature_values, bins=new_bins, labels=False, include_lowest=True)

    # Identify the indices where pd.cut returns NaN (these are the values outside the bins)
    nan_indices = cut_results.isna()

    # Step 2: Create outlier labels for values above and below the bin range
    outlier_labels = np.where(feature_values > new_bins.max(), n_bins,  # Assign n_bins for values above the max
                            np.where(feature_values < new_bins.min(), -1,  # Assign -1 for values below the min
                                    cut_results))

    outlier_condition = (outlier_labels == -1) | (outlier_labels == n_bins)

    cut_results[nan_indices & outlier_condition] = outlier_labels[nan_indices & outlier_condition]

    cut_results += 1

    return cut_results



def discretize_features(data_current_dt, rolling_data2, n_bins, current_date):

    # Extend the binning using different z-score extensions
    z_extensions = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 3, 5, 10]

    n_bins_init = n_bins - 2*len(z_extensions) - 2   # so the nb of bins is n_bins after extension of binning space

    volume_features = data_current_dt.columns[data_current_dt.columns.str.contains('BIDs|ASKs')]
    price_features = data_current_dt.columns[data_current_dt.columns.str.contains('BIDp|ASKp')]

    past_data = rolling_data2.loc[rolling_data2['date'] < current_date].copy(deep = True)
    past_data = past_data.dropna().sort_values(by=['sequence_number']).reset_index(drop = True)

    discrete_data = data_current_dt.copy(deep = True)


    # Create bins based on the past data for volume and price features
    flat_series_volume_ref = past_data[volume_features].values.flatten()
    _, bins_volume = pd.cut(flat_series_volume_ref, bins=n_bins_init, labels=False, retbins=True)

    flat_series_price_ref = past_data[price_features].values.flatten()
    _, bins_price = pd.cut(flat_series_price_ref, bins=n_bins_init, labels=False, retbins=True)

    # get extended bins
    new_bins_volume = extend_bins(flat_series_volume_ref, bins_volume, z_extensions)
    new_bins_price = extend_bins(flat_series_price_ref, bins_price, z_extensions)


    # Discretize the features within the current bucket
    for feature in volume_features:
        binned_feature = apply_binning_cut(discrete_data[feature], new_bins_volume, n_bins)
        discrete_data[feature] = binned_feature


    for feature in price_features:
        binned_feature = apply_binning_cut(discrete_data[feature], new_bins_price, n_bins)
        discrete_data[feature] = binned_feature


    features_names = volume_features.tolist() + price_features.tolist()

    nan_summary = discrete_data[features_names].isna().sum()
    print(current_date, nan_summary.sort_values().index[-1], nan_summary.max())

    discrete_data[features_names] /= n_bins

    return discrete_data



def create_numpy_vector(data, sequence_length, num_features):
    """
    Create 3D NumPy arrays (sequence_length, num_features) for time-series models.
    """
    data_np = data.values
    num_sequences = len(data_np) - sequence_length + 1
    X_np = np.lib.stride_tricks.sliding_window_view(data_np[:, :-1], sequence_length, axis=0)
    X_np = np.transpose(X_np, (0, 2, 1))
    X_np = np.expand_dims(X_np, axis=1)
    y_np = data_np[sequence_length - 1:, -1]
    return X_np.astype(np.float32), y_np.astype(np.int64)



def balanced_sampling_single_date(df, class_col='y', max_samples_per_class=100000):
    """
    Perform balanced random sampling of the dataset based on class.

    Parameters:
    df (pd.DataFrame): The input dataset with class column 'y'.
    class_col (str): The column representing class labels.
    max_samples_per_class (int): Maximum number of samples per class.

    Returns:
    pd.DataFrame: The balanced and sampled dataset.
    """

    sampled_dfs = []
    sampled_indices = []

    # Sort the dataframe by 'sequence_number' to maintain order
    df = df.sort_values(by=['sequence_number'])

    # Group by class within the single date
    class_groups = df.groupby(class_col)

    # Get the minimum count of samples for any class
    min_class_count = min(len(class_group) for _, class_group in class_groups)

    # The number of samples to draw for each class is the smaller of 5000 or min_class_count
    n_samples = min(min_class_count, max_samples_per_class)

    # Sample n_samples from each class
    for _, class_group in class_groups:
        sampled_group = class_group.sample(n=n_samples, random_state=1)  # Adjust the random_state for reproducibility
        sampled_dfs.append(sampled_group)
        sampled_indices.append(sampled_group.index)

    # Concatenate all sampled class groups into one balanced DataFrame
    sampled_df = pd.concat(sampled_dfs, ignore_index=False).sort_values(by=['sequence_number'])

    return sampled_df, pd.Index(np.concatenate(sampled_indices))



def get_1d_df(rolling_data_df, date):
    """ Function to get the rolling data on a specific date """
    data_date_dt = rolling_data_df.loc[rolling_data_df['date'] == date]
    data_date_dt = data_date_dt.dropna()
    data_date_dt = data_date_dt.reset_index(drop=True)
    data_date_dt = data_date_dt.sort_values('sequence_number')

    return data_date_dt


def save_data_per_date(X_data, y_data, sequence_numbers, save_path, dataset_type, current_date):
    """
    Save processed X and y data for each date in compressed format.
    """
    current_date = current_date.replace('-', '')
    np.savez_compressed(f"{save_path}/{dataset_type}_{current_date}.npz", X_data=X_data, y_data=y_data, sequence_numbers=sequence_numbers)






def fetch_and_preprocess_by_date(
    dic_dates: Dict[str, List[str]],
    config: Dict,
    full_path_save: str,
    symbol: str,
    T: int,
    pred_horizon: int
):
    """
    Fetch, preprocess, and save data incrementally for a specific symbol, T, and prediction horizon.

    Parameters:
    - dic_dates: Dictionary containing train, validation, and test dates.
    - config: Configuration parameters loaded from external file.
    - full_path_save: Path to save processed data.
    - symbol: The trading symbol to process (e.g., "BTC-USDT").
    - T: Number of time steps for the sequence.
    - pred_horizon: Prediction horizon for the target variable.
    """
    rolling_days_memory = []  # To store rolling data
    data_last_T_obs_prior_dt = pd.DataFrame()
    normalization_window = config["normalization_window"]
    bucket_size = config["bucket_size"]
    n_bins = config["n_bins"]
    table = config["table"]
    exchanges = config["exchanges"]
    path_sqlite = config["path_sqlite"]
    tick_size = config["tick_size"][symbol]
    type_mean = config["type_mean"]
    type_target = config["type_target"]

    train_dates = dic_dates['train_dates']
    val_dates = dic_dates['val_dates']
    test_dates = dic_dates['test_dates']
    dates = train_dates + val_dates + test_dates

    for current_date in dates:
        print(f"Processing {symbol} data for {current_date} (T={T}, pred_horizon={pred_horizon})")

        # Fetch data
        data_current = fetch_data_between_dates(
            table, [symbol], exchanges, current_date, current_date, path_sqlite
        )

        # Feature rearrangement and preprocessing
        data_current, feature_names = rearrange_features_names(data_current)
        data_current = get_log_volume(data_current)
        data_current = get_mid_px(data_current)

        # Rolling memory management
        rolling_days_memory.append(data_current)
        if len(rolling_days_memory) < 2:
            continue

        if len(pd.concat(rolling_days_memory[:-1]).iloc[:-bucket_size]) < normalization_window:
            print(f"Not enough data up to prior day to get the normalization window up to {bucket_size} steps before current day.")
            continue

        # Combine rolling data and normalize
        rolling_data = pd.concat(rolling_days_memory, ignore_index=True).sort_values(by="sequence_number")
        # Normalize price and volume features
        print('rolling')
        data_rolling = rolling_data[feature_names].rolling(window=normalization_window)
        data_mean = data_rolling.mean().ffill()
        data_std = data_rolling.std().ffill()
        rolling_data[feature_names] = (rolling_data[feature_names] - data_mean[feature_names]) / data_std[feature_names]

        # Compute target
        rolling_data2 = compute_price_diff_target(rolling_data, pred_horizon, type_target, tick_size, type_mean)

        # Only need current date now
        data_current_dt = get_1d_df(rolling_data2, current_date)

        if len(data_current_dt) < T:
            print(f"Not enough data for {current_date} to create X_np. Only {len(current_date)} obs vs T={T} for date {current_date}")
            continue  # Skip this date

        # Discretize features
        data_current_dt = discretize_features(data_current_dt, rolling_data2, n_bins, current_date)
        data_current_dt_vector = pd.concat([data_last_T_obs_prior_dt, data_current_dt.copy(deep = True)]).sort_values('sequence_number')
        data_last_T_obs_prior_dt = data_current_dt.iloc[-(T-1):].copy(deep = True) # for next iteration

        if len(data_current_dt) == len(data_current_dt_vector):
            print('No prior day discretized data yet')
            continue

        # keep track of keys for each training observations
        sequence_numbers = data_current_dt['sequence_number'].unique()

        # Build the matrix with (40 features, T time steps)
        X_np, y_np = create_numpy_vector(data_current_dt_vector[feature_names + ['y']], T, len(feature_names))

        if current_date in train_dates:
            print('sampling')
            sampled_df, sampled_indices = balanced_sampling_single_date(data_current_dt, 'y', max_samples_per_class=300000)
            print(sampled_df['y'].value_counts())

            sampled_indices_np = sampled_indices.to_numpy()
            sampled_indices_np.sort()

            X_np = X_np[sampled_indices_np]
            y_np = y_np[sampled_indices_np]

            sequence_numbers = sampled_df.loc[sampled_df.index.isin(sampled_indices_np)]['sequence_number'].unique()

        # Save processed data
        if current_date in train_dates:
            save_data_per_date(X_np, y_np, sequence_numbers, full_path_save, 'train', current_date)
        elif current_date in val_dates:
            save_data_per_date(X_np, y_np, sequence_numbers, full_path_save, 'val', current_date)
        elif current_date in test_dates:
            save_data_per_date(X_np, y_np, sequence_numbers, full_path_save, 'test', current_date)

        # Manage memory
        rolling_days_memory.pop(0)

        del data_current, data_current_dt, data_current_dt_vector, rolling_data, rolling_data2, data_rolling, data_mean, data_std
        del X_np, y_np, sequence_numbers
        gc.collect()

    del data_last_T_obs_prior_dt, rolling_days_memory
    gc.collect()

    print(f"Processing completed for {symbol} (T={T}, pred_horizon={pred_horizon}).")


def run_data_pipeline(config_path: str):
    """
    Main function to execute the data preprocessing pipeline for multiple symbols, T, and prediction horizon values.

    Parameters:
    - config_path: Path to the JSON configuration file.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Loop through symbols, T values, and prediction horizons
    for symbol in config["symbols"]:
        for T, pred_horizon in list(zip(config["Ts"], config["pred_horizons"])):
            print(f"Starting pipeline for {symbol} (T={T}, pred_horizon={pred_horizon})")

            # Generate study dates for the current symbol, T, and pred_horizon
            study_dates = get_dates(
                config["table"],
                [symbol],
                config["exchanges"],
                config["start_date"],
                config["end_date"],
                config["path_sqlite"],
                config["min_rows"]
            )

            train_dates, val_dates, test_dates = split_dates(study_dates, config["train_ratio"], config["val_ratio"])
            dic_dates = {"train_dates": train_dates, "val_dates": val_dates, "test_dates": test_dates}

            if config["rearrange_date_needed"]:
                train_dates, val_dates, test_dates, dic_dates = rearrange_dates(
                    dic_dates, config["val_date_st"], config["test_date_st"]
                )

            print(dic_dates)

            # Prepare save path for each combination
            folder_name = f"{symbol}_data_{dic_dates['train_dates'][0]}_{dic_dates['test_dates'][-1]}_T{T}_H{pred_horizon}"
            full_path_save = os.path.join(config["path_save_dataset"], folder_name)
            os.makedirs(full_path_save, exist_ok=True)

            # Process data for the current symbol, T, and pred_horizon
            fetch_and_preprocess_by_date(dic_dates, config, full_path_save, symbol, T, pred_horizon)

    print("Pipeline executed successfully for all configurations.")