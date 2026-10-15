import pandas as pd
import os

def load_data(path="data"):
    """
    Loads and merges chunked CSV files from a directory.

    Args:
        path (str): The path to the directory containing the CSV chunks.

    Returns:
        pd.DataFrame: A merged DataFrame containing data from all chunks.
    """
    # Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

    # Create a list to store the dataframes
    dfs = []

    # Iterate over the CSV files and append them to the list
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(path, csv_file))
        dfs.append(df)

    # Concatenate the dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df
