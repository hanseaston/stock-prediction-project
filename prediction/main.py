from data_pipeline.fetch_polygon import PolygonParser
from datetime import datetime

if __name__ == '__main__':
    # go through a list of stocks (implement in fetch API)
    parser = PolygonParser("../prediction/data")
    date_start = "2023-04-01"  # needs more efficien timplementation
    date_end = datetime.now().strftime("%Y-%m-%d")  # today
    parser.parse_sp500_tickers(date_start, date_end)
