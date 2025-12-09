import pandas as pd
import os

def merge_data(raw_data_path, processed_data_path):
    """
    Merges the raw train and store data and saves it to the processed data folder.

    Args:
        raw_data_path (str): The path to the raw data folder.
        processed_data_path (str): The path to the processed data folder.
    """
    train_csv_path = os.path.join(raw_data_path, 'train.csv')
    store_csv_path = os.path.join(raw_data_path, 'store.csv')
    merged_csv_path = os.path.join(processed_data_path, 'train_merged.csv')

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    try:
        train_df = pd.read_csv(train_csv_path, low_memory=False)
        store_df = pd.read_csv(store_csv_path)

        merged_df = pd.merge(train_df, store_df, on='Store', how='left')

        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Successfully merged data and saved to {merged_csv_path}")

    except FileNotFoundError as e:
        print(f"Error during data merging: {e}")
        print("Please ensure 'train.csv' and 'store.csv' are in the raw data directory.")

if __name__ == '__main__':
    # This allows the script to be run directly for data preparation
    # Example usage:
    # python src/data_prep.py
    from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
    merge_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)

