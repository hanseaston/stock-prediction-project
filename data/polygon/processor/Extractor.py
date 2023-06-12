import pandas as pd


def extract_all_data_from_file(file_path):
    return pd.read_csv(file_path)
