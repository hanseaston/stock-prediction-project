import pandas as pd
import os

from config import get_processed_dataset_path_from_name


class PolygonDataSelector:
    """
    Selects subsets of the data for each processed ticker data, based on user-defined conditions
    """

    def __init__(self, dataset_name):
        self.processed_dataset_path = get_processed_dataset_path_from_name(
            dataset_name)

    def select_data(self, file_name, date_start, date_end, skip_rows, column_names):
        processed_file_path = os.path.join(
            self.processed_dataset_path, file_name)
        df = pd.read_csv(processed_file_path)

        row_start = skip_rows + \
            0 if date_start is None else self.get_row_for_date(date_start)
        row_end = df.shape[0] - \
            1 if date_end is None else self.get_row_for_date(date_end)
        df = df.loc[row_start: row_end, column_names].reset_index(drop=True)
        return df

    def get_row_for_date(self, df, date):
        rows = df.loc[df['date'] == date]
        if len(rows) > 1:
            raise PolygonSelectorError(
                f"More than one entry found matching for date {date}")
        if len(rows) < 1:
            raise PolygonSelectorError(
                f"No entry found matching for date {date}")
        return rows.index[0]


class PolygonSelectorError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
