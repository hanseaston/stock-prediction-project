from data.polygon.processor.Processor import PolygonDataProcessor
from data.polygon.processor.Filterer import PolygonDataFilterer


if __name__ == '__main__':
    polygon_processor = PolygonDataProcessor(
        dataset_name='sp500')
    polygon_filterer = PolygonDataFilterer(dataset_name='sp500')
    polygon_processor.process_data()
    polygon_filterer.filter_data_by_num_entries()
