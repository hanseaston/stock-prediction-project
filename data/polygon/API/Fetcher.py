import requests


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

    def fetch(self, endpoint, params=None):
        url = self.base_url + endpoint
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print('Request failed with status code:', response.status_code)
            print('Requested endpoint is:', url)
            print('More detail:', response.json()['error'])
            raise ValueError()

    def fetch_from_next_url(self, response):
        response = requests.get(response['next_url'], headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            print('Request got next url failed with status code:',
                  response.status_code)
            return None

    def has_next_url(self, response):
        return 'next_url' in response
