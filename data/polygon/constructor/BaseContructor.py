from abc import ABC, abstractmethod

from data.polygon.constructor.Selector import PolygonDataSelector
from config import get_processed_dataset_path_from_name


class PolygonBaseConstructor(ABC):
    """
    Base dataset contructor that all constructors need to inherit from.
    """

    def __init__(self, dataset_name):
        self.processed_dataset_path = get_processed_dataset_path_from_name(
            dataset_name)
        self.selector = PolygonDataSelector(dataset_name)

    @abstractmethod
    def build_dataset_all_tickers(self):
        pass

    @abstractmethod
    def get_feature_dimension(self):
        pass
