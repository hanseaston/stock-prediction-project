import pandas as pd

import os

from config import PATHS

from data.polygon.processor.indicators.SMA import SMA
from data.polygon.processor.indicators.PrevDayChange import PrevDayChange


class PolygonDataProcessor:
    def __init__(self, dataset_name, append_mode):
        self.raw_dataset_path = self.get_processed_dataset_path_from_name(
            dataset_name) if append_mode else self.get_raw_dataset_path_from_name(dataset_name)
        self.processed_dataset_path = self.get_processed_dataset_path_from_name(
            dataset_name)

    def write_1_day_price_change(self, data):
        PrevDayChange(data, 'close_price', 'PDC_P1')

    def write_5_day_price_moving_average(self, data):
        SMA(data, 'close_price', 'SMA_P5', 5)

    def write_10_day_price_moving_average(self, data):
        SMA(data, 'close_price', 'SMA_P10', 10)

    def write_20_day_price_moving_average(self, data):
        SMA(data, 'close_price', 'SMA_P20', 20)

    def write_30_day_price_moving_average(self, data):
        SMA(data, 'close_price', 'SMA_P30', 30)

    def get_raw_dataset_path_from_name(self, dataset_name):
        if dataset_name == 'sp500':
            return PATHS['polygon_dataset_sp500_raw']
        elif dataset_name == 'nasdaq':
            return PATHS['polygon_dataset_nasdaq_raw']

    def get_processed_dataset_path_from_name(self, dataset_name):
        if dataset_name == 'sp500':
            return PATHS['polygon_dataset_sp500_processed']
        elif dataset_name == 'nasdaq':
            return PATHS['polygon_dataset_nasdaq_processed']

    def process_data(self):
        for file_name in os.listdir(self.raw_dataset_path):
            raw_file_path = os.path.join(self.raw_dataset_path, file_name)
            processed_file_path = os.path.join(
                self.processed_dataset_path, file_name)
            data = pd.read_csv(raw_file_path)
            self.write_5_day_price_moving_average(data)
            self.write_10_day_price_moving_average(data)
            self.write_20_day_price_moving_average(data)
            self.write_30_day_price_moving_average(data)
            self.write_1_day_price_change(data)
            data.to_csv(processed_file_path, index=False)
