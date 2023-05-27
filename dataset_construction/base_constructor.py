from abc import ABC, abstractmethod
import numpy as np
import csv
import os


class base_constructor(ABC):

    def __init__(self, data_source_dir):
        self.feature_dimen = None
        self.data_source_dir = data_source_dir
        self.training_ratio = 0.70
        self.validation_ratio = 0.15

    @abstractmethod
    def construct_model_dataset(self):
        pass

    def get_feature_dimension(self):
        if self.feature_dimen is None:
            raise KeyError("feature dimension is not set yet.")
        return self.feature_dimen

    def has_row_data_missing(self, row):
        for element in row:
            if not element.strip():
                return True
        return False

    def construct_data_matrix(self):
        with open(self.data_source_dir, 'r') as f:
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

    def construct_train_valid_test_set_from_X_y(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = X.shape[0]

        # training set, 70%, validation and test, 15%
        train_index = int(n * self.training_ratio)
        valid_index = int(n * (self.training_ratio + self.validation_ratio))

        train_X = X[0: train_index]
        train_y = y[0: train_index]
        valid_X = X[train_index: valid_index]
        valid_y = y[train_index: valid_index]
        test_X = X[valid_index:]
        test_Y = y[valid_index:]

        return train_X, train_y, valid_X, valid_y, test_X, test_Y

    def validate_dataset(self):
        for filename in os.listdir(self.data_source_dir):
            filepath = os.path.join(self.data_source_dir, filename)
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                rows = list(reader)
                current_row = 0
                for row in rows:
                    if current_row >= 30 and self.has_row_data_missing(row):
                        raise KeyError(
                            f'detect missing entries in row for {filename}')
