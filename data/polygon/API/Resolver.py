from tqdm import tqdm
import datetime
import csv

from data.polygon.API.Fetcher import PolygonAPIFetcher
from utils.utils import remove_all_files_from_dir


class PolygonAPIResolver:
    def __init__(self, data_base_url):
        self.polygon_api = PolygonAPIFetcher()
        self.data_file = data_base_url

    def parse_sp500_tickers(self, date_start, date_end):

        file_path = "../raw_data/sp500.csv"
        remove_all_files_from_dir(self.data_base_url)

        ticker_symbols = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                ticker_symbols.append(row[0])

        for i in tqdm(range(len(ticker_symbols))):
            ticker = ticker_symbols[i]
            self.parse_individual_ticker_within_time_range(
                ticker, date_start, date_end)

    def parse_nasdaq_tickers(self, date_start, date_end):

        nasdaq_path = "../raw_data/nasdaq.csv"
        remove_all_files_from_dir(self.data_base_url)

        nasdaq_ticker_symbols = []
        with open(nasdaq_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                nasdaq_ticker_symbols.append(row[0])

        EXCLUDE_SP500_SYMBOLS = True

        if EXCLUDE_SP500_SYMBOLS:
            # remove all stocks from sp500
            sp500_path = "../raw_data/sp500.csv"
            sp500_ticker_symbols = []
            with open(sp500_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    sp500_ticker_symbols.append(row[0])
            nasdaq_ticker_symbols = [
                ticker for ticker in nasdaq_ticker_symbols if ticker not in sp500_ticker_symbols]

        for ticker in nasdaq_ticker_symbols:
            self.parse_individual_ticker_within_time_range(
                ticker, date_start, date_end)

    def parse_tickers_from_stock_exchange(self):

        num_ticker_to_fetch = 1000
        date_start = "2020-06-01"
        date_end = "2019-12-11"
        stock_exchange = "XNYS"

        remove_all_files_from_dir(self.data_base_url)

        tickers_symbols = []

        response = self.polygon_api.fetch(
            f"v3/reference/tickers?type=CS&market=stocks&exchange={stock_exchange}&active=true")

        tickers_symbols.extend([obj['ticker'] for obj in response['results']])

        while len(tickers_symbols) < num_ticker_to_fetch and self.polygon_api.has_next_url(response):
            response = self.polygon_api.fetch_from_next_url(response)
            tickers_symbols.extend([obj['ticker']
                                   for obj in response['results']])

        for ticker in tickers_symbols:
            self.parse_individual_ticker_within_time_range(
                ticker, date_start, date_end)

    def parse_individual_ticker_within_time_range(self, ticker, date_start, date_end):

        response = self.polygon_api.fetch(
            f"v2/aggs/ticker/{ticker}/range/1/day/{date_start}/{date_end}?adjusted=true&sort=asc")

        if response is None or 'results' not in response:
            return

        ticker_history = response['results']
        ticker_name = response['ticker']

        file_name = f'{self.data_base_url}/{ticker_name}.csv'

        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['date', 'open_price', 'close_price',
                            'lowest_price', 'highest_price', 'volume'])

            for ticker_info in ticker_history:

                date = str(datetime.datetime.fromtimestamp(
                    ticker_info['t'] / 1000)).split(' ')[0]

                open_price = ticker_info['o']
                close_price = ticker_info['c']
                lowest_price = ticker_info['l']
                highest_price = ticker_info['h']
                volume = ticker_info['v']

                writer.writerow([date, open_price, close_price,
                                 lowest_price, highest_price, volume])
