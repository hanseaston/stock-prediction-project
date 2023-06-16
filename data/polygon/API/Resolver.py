from tqdm import tqdm
import csv
import os

from config import get_tickers_path_from_name, get_raw_dataset_path_from_name
from data.tickers.GetTickerNames import get_ticker_names

from data.polygon.API.Fetcher import PolygonAPIFetcher
from utils.utils import remove_all_files_from_dir


class PolygonAPIResolver:
    """
    Resolves user request, utilizes the Fetcher object,
    and returns user the constructed response
    """

    def __init__(self):
        self.polygon_fetcher = PolygonAPIFetcher()

    def resolve_sp500_dataset(self, date_start, date_end):
        """
        Fetches the SP&500 stocks ranging from date_start to date_end
        """

        sp500_symbols = get_ticker_names(get_tickers_path_from_name('sp500'))

        sp500_raw_data_path = get_raw_dataset_path_from_name('sp500')
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
                if ticker_data is not None:
                    writer.writerows(ticker_data)
