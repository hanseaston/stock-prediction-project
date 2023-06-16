import pandas as pd
import os

from config import get_processed_dataset_path_from_name


class PolygonDataSelector:

    def __init__(self, dataset_name):
        self.processed_dataset_path = get_processed_dataset_path_from_name(
            dataset_name)

    def select_data(self, file_name, date_start, date_end, skip_rows, column_names):
        processed_file_path = os.path.join(
            self.processed_dataset_path, file_name)
        if date_start is None and date_end is None:
            df = pd.read_csv(processed_file_path)
            df = df.loc[skip_rows:, column_names].reset_index(drop=True)
            return df
