import requests
import datetime
import csv


class PolygonAPI:
    def __init__(self):
        self.base_url = 'https://api.polygon.io/'
        self.headers = {
            'Authorization': 'Bearer cC8AtpZvOZlXEfTeQp7nI5oCxmePPQ7j',
        }

    def get(self, endpoint, params=None):
        url = self.base_url + endpoint
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print('Request failed with status code:', response.status_code)
            return None

    def get_from_next_url(self, response):
        response = requests.get(response['next_url'], headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('Request failed with status code:', response.status_code)
            return None

    def has_next_url(self, response):
        return response['next_url'] != None


class PolygonParser:
    def __init__(self):
        pass

    def parse_all_tickers(self):

        NUM_TICKERS_TO_FETCH = 1000

        polygon_api = PolygonAPI()

        tickers_symbols = []

        response = polygon_api.get(
            "v3/reference/tickers?market=stocks")
        tickers_symbols.extend([obj['ticker'] for obj in response['results']])

        while len(tickers_symbols) < NUM_TICKERS_TO_FETCH and polygon_api.has_next_url(response):
            print('here')
            response = polygon_api.get_from_next_url(response)
            tickers_symbols.extend([obj['ticker']
                                   for obj in response['results']])

        for ticker in tickers_symbols:
            response = polygon_api.get(
                f"v2/aggs/ticker/{ticker}/range/1/day/2021-06-01/2023-05-21?adjusted=true&sort=asc")
            polygon_parser.parse_individual_ticker_within_time_range(response)

    def parse_individual_ticker_within_time_range(self, response):
        ticker_history = response['results']
        ticker_name = response['ticker']

        file_name = f'./dataset/polygon/{ticker_name}.csv'

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


if __name__ == '__main__':
    polygon_parser = PolygonParser()
    polygon_parser.parse_all_tickers()
