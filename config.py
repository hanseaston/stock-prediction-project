import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    'nasdaq_ticker_symbols': os.path.join(ROOT_DIR, 'data', 'tickers', 'nasdaq.csv'),
    'sp500_ticker_symbols': os.path.join(ROOT_DIR, 'data', 'tickers', 'sp500.csv'),
    'polygon_dataset_sp500_raw': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'raw', 'sp500'),
    'polygon_dataset_nasdaq_raw': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'raw', 'nasdaq'),
    'polygon_dataset_sp500_processed': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'processed', 'sp500'),
    'polygon_dataset_nasdaq_processed': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'processed', 'nasdaq'),
    'polygon_dataset_sp500_ml': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'ml', 'sp500'),
    'polygon_dataset_nasdaq_ml': os.path.join(ROOT_DIR, 'data', 'polygon', 'dataset', 'ml', 'nasdaq'),
}


def get_trained_model_path_from_version(version_num, epoch_num):
    return os.path.join(ROOT_DIR, 'training', version_num, 'trained_models', f'{epoch_num}')


def get_tickers_path_from_name(dataset_name):
    if dataset_name == 'sp500':
        return PATHS['sp500_ticker_symbols']
    elif dataset_name == 'nasdaq':
        return PATHS['nasdaq_ticker_symbols']


def get_raw_dataset_path_from_name(dataset_name):
    if dataset_name == 'sp500':
        return PATHS['polygon_dataset_sp500_raw']
    elif dataset_name == 'nasdaq':
        return PATHS['polygon_dataset_nasdaq_raw']


def get_processed_dataset_path_from_name(dataset_name):
    if dataset_name == 'sp500':
        return PATHS['polygon_dataset_sp500_processed']
    elif dataset_name == 'nasdaq':
        return PATHS['polygon_dataset_nasdaq_processed']


def get_ml_dataset_path_from_name(dataset_name):
    if dataset_name == 'sp500':
        return PATHS['polygon_dataset_sp500_ml']
    elif dataset_name == 'nasdaq':
        return PATHS['polygon_dataset_nasdaq_ml']
