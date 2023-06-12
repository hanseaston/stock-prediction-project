from tqdm import tqdm
import csv
import os

from config import PATHS

from utils.utils import remove_all_files_from_dir, convert_date_to_yr
from data.polygon.API.Fetcher import PolygonAPIFetcher
from data.tickers.GetTickerNames import get_ticker_names


class PolygonAPIResolver:
    def __init__(self):
        self.polygon_fetcher = PolygonAPIFetcher()

    def resolve_sp500_dataset(self, date_start, date_end):

        sp500_symbols = get_ticker_names(PATHS['sp500_ticker_symbols'])

        year_start = convert_date_to_yr(date_start)
        year_end = convert_date_to_yr(date_end)

        sp500_raw_data_path = os.path.join(
            PATHS['polygon_dataset_raw'], f'sp500_{year_start}_{year_end}')

        os.makedirs(sp500_raw_data_path, exist_ok=True)
        remove_all_files_from_dir(sp500_raw_data_path)

        for i in tqdm(range(len(sp500_symbols))):

            ticker_name = sp500_symbols[i]
            ticker_file_path = os.path.join(
                sp500_raw_data_path, f'{ticker_name}.csv')

            with open(ticker_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['date', 'open_price', 'close_price',
                                 'lowest_price', 'highest_price', 'volume'])
                ticker_data = self.polygon_fetcher.fetch_data_in_range(
                    ticker_name, date_start, date_end)
                writer.writerows(ticker_data)
