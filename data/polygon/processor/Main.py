from data.polygon.processor.Processor import PolygonDataProcessor


if __name__ == '__main__':
    polygon_processor = PolygonDataProcessor(
        dataset_name='sp500', append_mode=False)
    polygon_processor.process_data()
