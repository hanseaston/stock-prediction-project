from base_constructor import base_constructor

import os
import numpy as np


class binary_constructor(base_constructor):

    def __init__(self, lag):
        super().__init__("../raw_data/polygon_processed")
        self.lag = lag
        self.trending_down_threshold = -0.01
        self.trending_up_threshold = 0.01

    def construct_model_dataset(self):

        X = []
        y = []

        for filename in os.listdir(self.data_source_dir):

            filepath = os.path.join(self.data_source_dir, filename)

            matrix = self.construct_data_matrix(filepath)

            n, d = matrix.shape

            self.feature_dimen = d

            # iterating each row for each stock ticker in the processed file
            for row_num in range(n):

                # if lag exceeds number of entries in the files, ignore
                if row_num + self.lag + 1 >= n:
                    break

                # if movement percentage being predicted (lag entries afterwards, 3 refers to the closing price
                # percentage) is too little, then ignore the training sample, otherwise, use the training sample
                percentage_change_after_lag = matrix[row_num + self.lag + 1][3]
                if self.is_percentage_between_thresholds(percentage_change_after_lag):
                    continue
                else:
                    X.append(matrix[row_num: row_num + self.lag])
                    y.append(self.convert_percentage_to_binary_label(
                        percentage_change_after_lag))

        return self.construct_train_valid_test_set_from_X_y(X, y)

    def is_percentage_between_thresholds(self, percentage):
        return percentage >= self.trending_down_threshold and percentage <= self.trending_up_threshold

    def convert_percentage_to_binary_label(self, percentage):
        return (np.sign(percentage) + 1) / 2
