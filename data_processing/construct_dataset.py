import csv
import os
import numpy as np


def is_percentage_between_thresholds(percentage):
    trending_down_threshold = -0.01
    trending_up_threshold = 0.01
    return percentage >= trending_down_threshold and percentage <= trending_up_threshold


def convert_percentage_to_one_hot_encoding(percentage):
    trending_down_threshold = -0.025
    trending_up_threshold = 0.025
    if percentage >= trending_up_threshold:
        return [0.0, 0.0, 1.0], 1
    elif percentage <= trending_down_threshold:
        return [1.0, 0.0, 0.0], -1
    else:
        return [0.0, 1.0, 0.0], 0


def convert_percentage_to_binary_label(percentage):
    return (np.sign(percentage) + 1) / 2


def has_row_data_missing(row):
    for element in row:
        if not element.strip():
            return True
    return False


def validate_dataset():
    path = '../dataset/polygon_processed'
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
            current_row = 0
            for row in rows:
                if current_row >= 30 and has_row_data_missing(row):
                    raise KeyError(
                        f'detect missing entries in row for {filename}')


def construct_data_matrix(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        stripped_rows = []
        line_number = 0
        for row in reader:
            # to remove the date info
            stripped_row = row[1:]
            # first 30 rows should always have entries missing (due to SMA_30 not having enough data yet)
            if line_number >= 30:
                stripped_rows.append(stripped_row)
            line_number += 1
        matrix = np.array(stripped_rows)
    return matrix.astype(float)


def construct_train_valid_test_set_from_X_y(X, y):
    X = np.array(X)
    y = np.array(y)
    n = X.shape[0]

    # training set, 70%, validation and test, 15%
    train_index = int(n * 0.70)
    valid_index = int(n * 0.85)

    train_X = X[0: train_index]
    train_y = y[0: train_index]
    valid_X = X[train_index: valid_index]
    valid_y = y[train_index: valid_index]
    test_X = X[valid_index:]
    test_Y = y[valid_index:]

    return train_X, train_y, valid_X, valid_y, test_X, test_Y


def construct_trinery_classification_dataset(lag):

    X = []
    y = []

    path = '../dataset/polygon_processed'

    trend_down_cnt = 0
    trend_even_cnt = 0
    trend_up_cnt = 0

    for filename in os.listdir(path):

        filepath = os.path.join(path, filename)

        matrix = construct_data_matrix(filepath)

        n, _ = matrix.shape

        # iterating each row for each stock ticker in the processed file
        for row_num in range(n):

            # if lag exceeds number of entries in the files, ignore
            if row_num + lag + 1 >= n:
                break

            percentage_change_after_lag = matrix[row_num + lag + 1][3]

            X.append(matrix[row_num: row_num + lag])

            one_hot_label, movement_direction = convert_percentage_to_one_hot_encoding(
                percentage_change_after_lag)

            # count label distribution
            if movement_direction == -1:
                trend_down_cnt += 1
            elif movement_direction == 0:
                trend_even_cnt += 1
            else:
                trend_up_cnt += 1

            y.append(one_hot_label)

    print(
        f"trend down count: {trend_down_cnt}, trend even count: {trend_even_cnt}, trend up count: {trend_up_cnt}")

    train_X, train_y, valid_X, valid_y, test_X, test_Y = construct_train_valid_test_set_from_X_y(
        X, y)

    # construct ratio for the labels
    sum = trend_down_cnt + trend_even_cnt + trend_up_cnt
    trend_down_ratio = trend_down_cnt / sum
    trend_even_ratio = trend_even_cnt / sum
    trend_up_ratio = trend_up_cnt / sum

    return train_X, train_y, valid_X, valid_y, test_X, test_Y, trend_down_ratio, trend_even_ratio, trend_up_ratio


def construct_binary_classification_dataset(lag):

    X = []
    y = []

    path = '../dataset/polygon_processed'

    for filename in os.listdir(path):

        filepath = os.path.join(path, filename)

        matrix = construct_data_matrix(filepath)

        n, _ = matrix.shape

        # iterating each row for each stock ticker in the processed file
        for row_num in range(n):

            # if lag exceeds number of entries in the files, ignore
            if row_num + lag + 1 >= n:
                break

            # if movement percentage being predicted (lag entries afterwards, 3 refers to the closing price
            # percentage) is too little, then ignore the training sample, otherwise, use the training sample
            percentage_change_after_lag = matrix[row_num + lag + 1][3]
            if is_percentage_between_thresholds(percentage_change_after_lag):
                continue
            else:
                X.append(matrix[row_num: row_num + lag])
                y.append(convert_percentage_to_binary_label(
                    percentage_change_after_lag))

    return construct_train_valid_test_set_from_X_y(X, y)


if __name__ == '__main__':
    train_X, train_y, valid_X, valid_y, test_X, test_Y = construct_trinery_classification_dataset(
        10)
