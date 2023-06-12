import requests
from datetime import datetime


class PolygonAPIFetcher:
    """
    Fetches data from Polygon API endpoints.
    Refer to https://polygon.io/ for documentation.
    """

    def __init__(self):
        self.base_url = 'https://api.polygon.io/'
        self.headers = {
            'Authorization': 'Bearer cC8AtpZvOZlXEfTeQp7nI5oCxmePPQ7j',
        }

    def fetch_data_in_range(self, ticker_name, date_start, date_end):

        response = self.fetch(
            f"v2/aggs/ticker/{ticker_name}/range/1/day/{date_start}/{date_end}?adjusted=true&sort=asc")

        if response is None or 'results' not in response:
            print('Response or the result is not defined', response)
            raise PolygonAPIError()

        ticker_history = response['results']
        ticker_data = []

        for ticker_info in ticker_history:

            date = str(datetime.fromtimestamp(
                ticker_info['t'] / 1000)).split(' ')[0]

            open_price = ticker_info['o']
            close_price = ticker_info['c']
            lowest_price = ticker_info['l']
            highest_price = ticker_info['h']
            volume = ticker_info['v']
            ticker_data.append(
                [date, open_price, close_price, lowest_price, highest_price, volume])

        return ticker_data

    def fetch(self, endpoint, params=None):
        url = self.base_url + endpoint
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print('Request failed with status code:', response.status_code)
            print('Requested endpoint is:', url)
            print('More detail:', response.json()['error'])
            raise PolygonAPIError()

    def fetch_from_next_url(self, response):
        response = requests.get(response['next_url'], headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('Request got next url failed with status code:',
                  response.status_code)
            raise PolygonAPIError()

    def has_next_url(self, response):
        return 'next_url' in response


class PolygonAPIError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
