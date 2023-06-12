import csv


def get_ticker_names(file_path):
    with open(file_path, 'r') as file:
        ticker_symbols = []
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            ticker_symbols.append(row[0])
        return ticker_symbols
