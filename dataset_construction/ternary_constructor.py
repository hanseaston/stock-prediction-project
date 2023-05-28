import os
import tensorflow as tf
import numpy as np


from dataset_construction.base_constructor import base_constructor
from utils.utils import check_all_zeros


class ternary_constructor(base_constructor):

    def __init__(self, lag):
        super().__init__("../raw_data/polygon_processed")
        self.lag = lag
        self.trending_down_threshold = -0.025
        self.trending_up_threshold = 0.025
        self.trend_down_cnt = 0
        self.initialize_trending_cnt()

    def construct_model_dataset(self):

        self.initialize_trending_cnt()

        X = []
        y = []

        for filename in os.listdir(self.data_source_dir):

            matrix = self.construct_data_matrix(filename)

            n, d = matrix.shape
            self.feature_dimen = d

            # iterating each row for each stock ticker in the processed file
            for row_num in range(n):

                # if lag exceeds number of entries in the files, ignore
                if row_num + self.lag + 1 >= n:
                    break

                percentage_change_after_lag = matrix[row_num + self.lag + 1][3]

                X.append(matrix[row_num: row_num + self.lag])

                movement_label = self.convert_percentage_to_label(
                    percentage_change_after_lag)

                # count label distribution
                if movement_label == 0:
                    self.trend_down_cnt += 1
                elif movement_label == 1:
                    self.trend_even_cnt += 1
                else:
                    self.trend_up_cnt += 1

                y.append(movement_label)

        return self.construct_train_valid_test_set_from_X_y(X, y)

    def get_trend_ratios(self):
        self.validate_trending_cnt()
        trend_down_ratio = self.trend_even_cnt / self.trend_down_cnt
        trend_up_ratio = self.trend_even_cnt / self.trend_up_cnt
        return tf.constant([trend_down_ratio, 1.0, trend_up_ratio])

    def convert_percentage_to_one_hot_encoding(self, percentage):
        if percentage >= self.trending_up_threshold:
            return tf.constant([0.0, 0.0, 1.0]), 1
        elif percentage <= self.trending_down_threshold:
            return tf.constant([1.0, 0.0, 0.0]), -1
        else:
            return tf.constant([0.0, 1.0, 0.0]), 0

    def convert_percentage_to_label(self, percentage):
        if percentage >= self.trending_up_threshold:
            return 2
        elif percentage <= self.trending_down_threshold:
            return 0
        else:
            return 1

    def convert_prediction_to_one_hot_encoding(self, predictions):
        max_indices = tf.argmax(predictions, axis=1)
        one_hot_predictions = tf.one_hot(max_indices, depth=3)
        return one_hot_predictions

    def initialize_trending_cnt(self):
        self.trend_down_cnt = 0
        self.trend_even_cnt = 0
        self.trend_up_cnt = 0

    def validate_trending_cnt(self):
        if check_all_zeros([self.trend_down_cnt, self.trend_even_cnt, self.trend_up_cnt]):
            raise ValueError("counts have not been intialized yet")
        if self.trend_even_cnt < self.trend_down_cnt or self.trend_even_cnt < self.trend_up_cnt:
            raise ValueError(
                "mistakes made when counting, trending even count should be the largest")
