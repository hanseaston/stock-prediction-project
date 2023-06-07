import numpy as np
import tensorflow as tf
import os
import csv
from utils.utils import get_file_name
from dataset_construction.binary_constructor import binary_constructor

np.set_printoptions(suppress=True)

# sp500
neutral_threshold_model_path = "../training/results/sp500/neutral/trained_model"
positive_threshold_model_path = "../training/results/sp500/positive/trained_model"
negative_threshold_model_path = "../training/results/sp500/negative/trained_model"
lag = 10


def is_better_prediction(pred1, pred2):
    neu1, pos1, neg1 = pred1[0], pred1[1], pred1[2]
    neu2, pos2, neg2 = pred2[0], pred2[1], pred2[2]
    if neu1 != neu2:
        return neu1 > neu2
    if pos1 != pos2:
        return pos1 > pos2
    return neg1 < neg2


def find_row_for_date(file_path, trading_start_date):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=0):
            if trading_start_date in row:
                if row_number < 50:
                    return -1
                return row_number
        return -1


def get_dates_between_range(trading_start_date, trading_end_date):
    file_path = '../raw_data/sp500_2014_2023_processed/AMCR.csv'
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        row_start = find_row_for_date(file_path, trading_start_date)
        row_end = find_row_for_date(file_path, trading_end_date)
        # get the dates, which is the first col
        return list(np.array(rows)[row_start: row_end + 1, 0])


def construct_input_data(file_path, starting_index):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        input = np.array(
            rows[starting_index - lag: starting_index])[:, 2:].astype(float)
        return input.reshape(1, input.shape[0], input.shape[1])


if __name__ == '__main__':

    processed_data_path = '../raw_data/sp500_2014_2023_processed'

    constructor = binary_constructor(10, 0.0, processed_data_path)
    testX = constructor.construct_dataset_for_agent()
    print(testX)

    # amount of money I have in the account
    money_in_account = 5000
    trading_start_date = '2022-06-15'
    trading_end_date = '2022-07-04'
    dates = get_dates_between_range(trading_start_date, trading_end_date)

    # model
    neutral_model = tf.keras.models.load_model(
        neutral_threshold_model_path)

    # prediction_threshold
    neutral_model_threshold = 0.9

    total = 0
    overall_winning = 0

    for date in dates:

        # need to keep track pf this index
        # since the row where the start date is different for each file
        ticker_file_start_index = []

        for ticker_file in sorted(os.listdir(processed_data_path)):
            file_path = os.path.join(processed_data_path, ticker_file)
            start_idx = find_row_for_date(
                file_path, date)
            ticker_file_start_index.append(start_idx)

        index = 0
        stocks_to_buy = []

        for ticker_file in sorted(os.listdir(processed_data_path)):

            # get file path
            file_path = os.path.join(processed_data_path, ticker_file)

            # if valid stock candidate
            if ticker_file_start_index[index] != -1:

                # construct data
                input_data = construct_input_data(
                    file_path, ticker_file_start_index[index])

                neu_p = neutral_model(
                    input_data, training=False).numpy()[0][0]

                if neu_p < neutral_model_threshold:
                    index += 1
                    continue

                stocks_to_buy.append([ticker_file, neu_p])

            index += 1

        if len(stocks_to_buy) == 0:
            continue

        losing_stocks_cnt = 0
        winning_stocks_cnt = 0
        best_stock_is_winning = None
        for stock_info_pair in stocks_to_buy:
            stock_to_buy = stock_info_pair[0]
            confidence_score = stock_info_pair[1]
            file_path = os.path.join(processed_data_path, stock_to_buy)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
                row_index = find_row_for_date(file_path, date)
                tomorrow_price = float(rows[row_index][1])
                today_price = float(rows[row_index - 1][1])

                if tomorrow_price < today_price:
                    losing_stocks_cnt += 1
                else:
                    winning_stocks_cnt += 1

                shared_to_buy = (money_in_account *
                                 confidence_score) // today_price
                money_in_account -= shared_to_buy * today_price
                money_in_account += shared_to_buy * tomorrow_price

        print(f"On day {date}, winning stocks: {winning_stocks_cnt}, lossing stocks: {losing_stocks_cnt}, ratio is {winning_stocks_cnt / len(stocks_to_buy)}")
        total += len(stocks_to_buy)
        overall_winning += winning_stocks_cnt
        print(
            f'{total} predictions made so far: {total}. Acc is {overall_winning / total}')
