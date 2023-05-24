import csv
import os


def is_approximately_zero(value, tolerance=1e-5):
    return abs(value) < tolerance


def check_invalid_data(attributes):
    for attribute in attributes:
        if is_approximately_zero(attribute):
            return True
    return False


def process_data(rows, filename):

    file_dir = f'./dataset/polygon_processed/{filename}'
    with open(file_dir, 'w', newline='') as f:
        writer = csv.writer(f)
        next(rows)

        previous_day_close = None
        previous_day_volume = None

        price_5_day_hist = []
        price_5_day_running_sum = 0.0

        price_10_day_hist = []
        price_10_day_running_sum = 0.0

        price_20_day_hist = []
        price_20_day_running_sum = 0.0

        price_30_day_hist = []
        price_30_day_running_sum = 0.0

        volume_5_day_hist = []
        volume_5_day_running_sum = 0.0

        volume_10_day_hist = []
        volume_10_day_running_sum = 0.0

        volume_20_day_hist = []
        volume_20_day_running_sum = 0.0

        volume_30_day_hist = []
        volume_30_day_running_sum = 0.0

        writer.writerow(['date', 'c_open', 'c_high',
                         'c_low', 'p_price', 'p_volume',
                         'p_5_SMA', 'vol_5_SMA', 'p_10_SMA',
                         'vol_10_SMA', 'p_20_SMA', 'vol_20_SMA',
                         'p_30_SMA', 'vol_30_SMA'])

        for row in rows:
            date = str(row[0])

            open_price = float(row[1])
            close_price = float(row[2])
            lowest_price = float(row[3])
            highest_price = float(row[4])
            volume = float(row[5])

            if check_invalid_data([open_price, close_price, lowest_price, highest_price, volume]):
                print(f'Removing {file_dir} due to invalid data')
                os.remove(file_dir)
                return

            c_open = open_price / close_price - 1
            c_high = highest_price / close_price - 1
            c_low = lowest_price / close_price - 1
            p_price = None if previous_day_close is None else previous_day_close / close_price - 1
            p_volume = None if previous_day_volume is None else previous_day_volume / volume - 1
            previous_day_close = close_price
            previous_day_volume = volume

            price_5_day_moving_average = None
            volume_5_day_moving_average = None
            price_10_day_moving_average = None
            volume_10_day_moving_average = None
            price_20_day_moving_average = None
            volume_20_day_moving_average = None
            price_30_day_moving_average = None
            volume_30_day_moving_average = None

            if len(price_5_day_hist) < 5:
                price_5_day_hist.append(close_price)
                price_5_day_running_sum += close_price
            else:
                price_5_day_moving_average = price_5_day_running_sum / 5 / close_price - 1
                earliest_price = price_5_day_hist.pop(0)
                price_5_day_hist.append(close_price)
                price_5_day_running_sum = price_5_day_running_sum - earliest_price + close_price

            if len(volume_5_day_hist) < 5:
                volume_5_day_hist.append(volume)
                volume_5_day_running_sum += volume
            else:
                volume_5_day_moving_average = volume_5_day_running_sum / 5 / volume - 1
                earliest_volume = volume_5_day_hist.pop(0)
                volume_5_day_hist.append(volume)
                volume_5_day_running_sum = volume_5_day_running_sum - earliest_volume + volume

            if len(price_10_day_hist) < 10:
                price_10_day_hist.append(close_price)
                price_10_day_running_sum += close_price
            else:
                price_10_day_moving_average = price_10_day_running_sum / 10 / close_price - 1
                earliest_price = price_10_day_hist.pop(0)
                price_10_day_hist.append(close_price)
                price_10_day_running_sum = price_10_day_running_sum - earliest_price + close_price

            if len(volume_10_day_hist) < 10:
                volume_10_day_hist.append(volume)
                volume_10_day_running_sum += volume
            else:
                volume_10_day_moving_average = volume_10_day_running_sum / 10 / volume - 1
                earliest_volume = volume_10_day_hist.pop(0)
                volume_10_day_hist.append(volume)
                volume_10_day_running_sum = volume_10_day_running_sum - earliest_volume + volume

            if len(price_20_day_hist) < 20:
                price_20_day_hist.append(close_price)
                price_20_day_running_sum += close_price
            else:
                price_20_day_moving_average = price_20_day_running_sum / 20 / close_price - 1
                earliest_price = price_20_day_hist.pop(0)
                price_20_day_hist.append(close_price)
                price_20_day_running_sum = price_20_day_running_sum - earliest_price + close_price

            if len(volume_20_day_hist) < 20:
                volume_20_day_hist.append(volume)
                volume_20_day_running_sum += volume
            else:
                volume_20_day_moving_average = volume_20_day_running_sum / 20 / volume - 1
                earliest_volume = volume_20_day_hist.pop(0)
                volume_20_day_hist.append(volume)
                volume_20_day_running_sum = volume_20_day_running_sum - earliest_volume + volume

            if len(price_30_day_hist) < 30:
                price_30_day_hist.append(close_price)
                price_30_day_running_sum += close_price
            else:
                price_30_day_moving_average = price_30_day_running_sum / 30 / close_price - 1
                earliest_price = price_30_day_hist.pop(0)
                price_30_day_hist.append(close_price)
                price_30_day_running_sum = price_30_day_running_sum - earliest_price + close_price

            if len(volume_30_day_hist) < 30:
                volume_30_day_hist.append(volume)
                volume_30_day_running_sum += volume
            else:
                volume_30_day_moving_average = volume_30_day_running_sum / 30 / volume - 1
                earliest_volume = volume_30_day_hist.pop(0)
                volume_30_day_hist.append(volume)
                volume_30_day_running_sum = volume_30_day_running_sum - earliest_volume + volume

            writer.writerow([date, c_open, c_high, c_low, p_price,
                             p_volume, price_5_day_moving_average,
                             volume_5_day_moving_average, price_10_day_moving_average,
                             volume_10_day_moving_average, price_20_day_moving_average,
                             volume_20_day_moving_average, price_30_day_moving_average,
                             volume_30_day_moving_average])


if __name__ == '__main__':
    # specify your path
    path = './dataset/polygon'

    # iterate over files in that directory
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                process_data(reader, filename)
