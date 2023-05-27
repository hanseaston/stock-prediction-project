from base_constructor import base_constructor


import os
from ..utils.utils import check_all_zeros


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

        for filename in os.listdir(self.bas):

            filepath = os.path.join(self.data_source_dir, filename)

            matrix = self.construct_data_matrix(filepath)

            n, d = matrix.shape
            self.feature_dimen = d

            # iterating each row for each stock ticker in the processed file
            for row_num in range(n):

                # if lag exceeds number of entries in the files, ignore
                if row_num + self.lag + 1 >= n:
                    break

                percentage_change_after_lag = matrix[row_num + self.lag + 1][3]

                X.append(matrix[row_num: row_num + self.lag])

                one_hot_label, movement_direction = self.convert_percentage_to_one_hot_encoding(
                    percentage_change_after_lag)

                # count label distribution
                if movement_direction == -1:
                    self.trend_down_cnt += 1
                elif movement_direction == 0:
                    self.trend_even_cnt += 1
                else:
                    self.trend_up_cnt += 1

                y.append(one_hot_label)

        return self.construct_train_valid_test_set_from_X_y(X, y)

    def get_trend_cnt_ratio(self):
        self.validate_trending_cnt()
        sum = self.trend_down_cnt + self.trend_even_cnt + self.trend_up_cnt
        trend_down_ratio = self.trend_down_cnt / sum
        trend_even_ratio = self.trend_even_cnt / sum
        trend_up_ratio = self.trend_up_cnt / sum
        return trend_down_ratio, trend_even_ratio, trend_up_ratio

    def convert_percentage_to_one_hot_encoding(self, percentage):
        if percentage >= self.trending_up_threshold:
            return [0.0, 0.0, 1.0], 1
        elif percentage <= self.trending_down_threshold:
            return [1.0, 0.0, 0.0], -1
        else:
            return [0.0, 1.0, 0.0], 0

    def initialize_trending_cnt(self):
        self.trend_down_cnt = 0
        self.trend_even_cnt = 0
        self.trend_up_cnt = 0

    def validate_trending_cnt(self):
        if check_all_zeros([self.trend_down_cnt, self.trend_even_cnt, self.trend_up_cnt]):
            raise ValueError("counts have not been intialized yet")
