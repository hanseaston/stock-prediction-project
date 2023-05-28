from dataset_construction.base_constructor import base_constructor

import os
import numpy as np
from imblearn.over_sampling import SMOTE


class binary_constructor(base_constructor):

    def __init__(self, lag):
        super().__init__("../raw_data/polygon_processed")
        self.lag = lag
        self.threshold = 0.00

    def construct_model_dataset(self):

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
                y.append(self.convert_percentage_to_binary_label(
                    percentage_change_after_lag))

        # initialize into numpy array
        X = np.array(X)
        y = np.array(y)

        # random shuffling by indices
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        train_X, train_y, valid_X, valid_y, test_X, test_y = self.construct_train_valid_test_set_from_X_y(
            X, y)

        print(np.count_nonzero(train_y == 0))
        print(np.count_nonzero(train_y == 1))

        return train_X, train_y, valid_X, valid_y, test_X, test_y

        X_flat = train_X.reshape(train_X.shape[0], -1)

        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_flat, train_y)

        X_train_resampled = X_train_resampled.reshape(
            X_train_resampled.shape[0], train_X.shape[1], train_X.shape[2])

        return X_train_resampled, y_train_resampled, valid_X, valid_y, test_X, test_y

        # augment negative examples
        negative_examples_cnt = np.count_nonzero(train_y == 0)
        positive_examples_cnt = np.count_nonzero(train_y == 1)

        ratio = negative_examples_cnt / positive_examples_cnt

        X_positive_sample = train_X[np.where(train_y == 1)[0]]
        y_positive_sample = train_y[np.where(train_y == 1)[0]]

        X_negative_sample = train_X[np.where(train_y == 0)[0]]
        y_negative_sample = train_y[np.where(train_y == 0)[0]]

        X_positive_sample = np.repeat(X_positive_sample, repeats=ratio, axis=0)
        y_positive_example = np.repeat(
            y_positive_sample, repeats=ratio, axis=0)

        train_X = np.concatenate(
            (X_positive_sample, X_negative_sample), axis=0)
        train_y = np.concatenate(
            (y_positive_example, y_negative_sample), axis=0)

        print(np.count_nonzero(train_y == 0))
        print(np.count_nonzero(train_y == 1))

        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]

        return train_X, train_y, valid_X, valid_y, test_X, test_y

    def convert_percentage_to_binary_label(self, percentage):
        return 1 if percentage >= self.threshold else 0
