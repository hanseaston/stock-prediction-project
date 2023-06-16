import os
import pandas as pd

from config import get_processed_dataset_path_from_name

from utils.utils import get_file_name


class PolygonDataFilterer:
    """
    Filter out ticker data on individual ticker level.
    """

    def __init__(self, dataset_name):
        self.processed_dataset_path = get_processed_dataset_path_from_name(
            dataset_name)

    def filter_data_by_num_entries(self):
        """
        Removes all ticker files that contain less entries than the data range 
        specified
        """

        # calculates the most number of entries of all processed ticker files
        max_num_rows = -1
        for file_name in os.listdir(self.processed_dataset_path):
            processed_file_path = os.path.join(
                self.processed_dataset_path, file_name)
            data = pd.read_csv(processed_file_path)
            num_rows = data.shape[0]
            max_num_rows = max(max_num_rows, num_rows)

        # removes and reports all that contain less than the max number of entries
        files_removed = set()
        for file_name in os.listdir(self.processed_dataset_path):
            processed_file_path = os.path.join(
                self.processed_dataset_path, file_name)
            data = pd.read_csv(processed_file_path)
            num_rows = data.shape[0]
            if num_rows < max_num_rows:
                os.remove(processed_file_path)
                files_removed.add(get_file_name(file_name))
        print(
            f"Total number of stocks removed: {len(files_removed)}. The removed tickers are {files_removed}")
