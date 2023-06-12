import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    'nasdaq_ticker_symbols': os.path.join(ROOT_DIR, 'data', 'tickers', 'nasdaq.csv'),
    'sp500_ticker_symbols': os.path.join(ROOT_DIR, 'data', 'tickers', 'sp500.csv'),
    'polygon_dataset_sp500_raw': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'raw', 'sp500'),
    'polygon_dataset_nasdaq_raw': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'raw', 'nasdaq'),
    'polygon_dataset_sp500_processed': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'processed', 'sp500'),
    'polygon_dataset_nasdaq_processed': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'processed', 'nasdaq'),
}
