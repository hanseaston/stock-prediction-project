from abc import ABC, abstractmethod
import numpy as np

from data.polygon.constructor.Selector import PolygonDataSelector
from config import get_processed_dataset_path_from_name


class PolygonBaseConstructor(ABC):

    def __init__(self, dataset_name):
        self.processed_dataset_path = get_processed_dataset_path_from_name(
            dataset_name)
        self.selector = PolygonDataSelector(dataset_name)

    @abstractmethod
    def build_dataset_all_tickers(self):
        pass

    @abstractmethod
    def build_dataset_single_ticker(self, df):
        pass

    @abstractmethod
    def get_feature_dimension(self):
        pass

    def shuffle_dataset(self, train_X, train_y, valid_X, valid_y, test_X, test_y):
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        valid_X = np.array(valid_X)
        valid_y = np.array(valid_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]
        return train_X, train_y, valid_X, valid_y, test_X, test_y
