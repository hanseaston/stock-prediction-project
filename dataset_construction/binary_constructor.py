from dataset_construction.base_constructor import base_constructor

import os
import numpy as np
import csv


class binary_constructor(base_constructor):

    def __init__(self, lag, threshold, data_path):
        super().__init__(data_path)
        self.lag = lag
        self.threshold = threshold
        self.outlier_threshold = 0.05

    def construct_prediction_dataset(self):
        """
        This method constructs the dataset for deployment (sorta)
        It assumes the stock data for today is already in the file (by running the data pipeline)
        And then take the rows[-self.lag:] (basically the most recent X days) of data
        Probably not gonna use it unless we are ready for deployment
        """

        X = []
        for filename in os.listdir(self.data_source_dir):
            filepath = os.path.join(self.data_source_dir, filename)
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                rows = [row[1:] for row in reader]  # remove date information
                rows = np.array(rows[-self.lag:])
                X.append(rows)
        X = np.array(X).astype(float)
        return X

    def construct_evaluation_dataset(self):
        """
        This method parses the dataset for evaluation (NOT training)
        Probably has lots of redundacy (need to simplify)
        """

        X = []
        y = []

        for filename in os.listdir(self.data_source_dir):

            matrix = self.construct_data_matrix(filename, True)

            n, d = matrix.shape

            self.feature_dimen = d

            # iterating each row for each stock ticker in the processed file
            for row_num in range(n):

                # if lag exceeds number of entries in the files, ignore
                if row_num + self.lag >= n:
                    break

                percentage_change_after_lag = matrix[row_num + self.lag][3]
                X.append(matrix[row_num: row_num + self.lag])
                y.append(self.convert_percentage_to_binary_label(
                    percentage_change_after_lag))

        # X, y = self.remove_outliers(X, y, self.outlier_threshold)

        # initialize into numpy array
        X = np.array(X)
        y = np.array(y)

        return X, y

    def construct_model_dataset_v2(self):
        """
        Probably needs to change the name
        This is the "correct" way of doing data split (by split each file's timeline)
        And after using this model, the model performs a lot worse (SAD)
        """

        train_X = []
        train_y = []
        valid_X = []
        valid_y = []
        test_X = []
        test_y = []

        for filename in os.listdir(self.data_source_dir):

            matrix = self.construct_data_matrix(
                filename, True)

            n, d = matrix.shape

            self.feature_dimen = d

            # Specifies the range of the train, validation, and test set
            train_index = int(n * self.training_ratio)
            valid_index = int(
                n * (self.training_ratio + self.validation_ratio))

            # iterating each row for each stock ticker in the processed file
            for row_num in range(n):

                # if lag exceeds number of entries in the files, ignore
                if row_num + self.lag >= n:
                    break

                percentage_change_after_lag = matrix[row_num + self.lag][3]

                # Depending on the row number, put into train, validation, or test
                if row_num <= train_index:
                    train_X.append(matrix[row_num: row_num + self.lag])
                    train_y.append(self.convert_percentage_to_binary_label(
                        percentage_change_after_lag))
                elif row_num <= valid_index:
                    valid_X.append(matrix[row_num: row_num + self.lag])
                    valid_y.append(self.convert_percentage_to_binary_label(
                        percentage_change_after_lag))
                else:
                    test_X.append(matrix[row_num: row_num + self.lag])
                    test_y.append(self.convert_percentage_to_binary_label(
                        percentage_change_after_lag))

        # intialize into numpy array
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        valid_X = np.array(valid_X)
        valid_y = np.array(valid_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)

        # shuffling
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]

        ###### THE CODE BELOW ONLY APPLIES IF DATA NEEDS TO BE UPSAMPLED OR DOWNSAMPLED #####

        # Can print out the labels to see if there is an imbalance
        negative_examples_cnt = np.count_nonzero(train_y == 0)
        positive_examples_cnt = np.count_nonzero(train_y == 1)
        print(
            f'Positive count: {positive_examples_cnt}. Negative count: {negative_examples_cnt} ', )

        # No need to oversample since threshold is balanced (AT LEAST TRUE FOR SP500)
        if self.threshold == 0.0:
            return train_X, train_y, valid_X, valid_y, test_X, test_y

        # augment negative examples when threshold is not 0 (thus classification is imbalanced)
        negative_examples_cnt = np.count_nonzero(train_y == 0)
        positive_examples_cnt = np.count_nonzero(train_y == 1)

        ratio = negative_examples_cnt / positive_examples_cnt

        X_positive_sample = train_X[np.where(train_y == 1)[0]]
        y_positive_sample = train_y[np.where(train_y == 1)[0]]

        X_negative_sample = train_X[np.where(train_y == 0)[0]]
        y_negative_sample = train_y[np.where(train_y == 0)[0]]

        # TODO: currently we are assuming the positive samples are undersampled
        X_positive_sample = np.repeat(X_positive_sample, repeats=ratio, axis=0)
        y_positive_example = np.repeat(
            y_positive_sample, repeats=ratio, axis=0)

        train_X = np.concatenate(
            (X_positive_sample, X_negative_sample), axis=0)
        train_y = np.concatenate(
            (y_positive_example, y_negative_sample), axis=0)

        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]

        negative_examples_cnt = np.count_nonzero(train_y == 0)
        positive_examples_cnt = np.count_nonzero(train_y == 1)

        # Can print out the adjusted count after resampling
        # print(positive_examples_cnt, negative_examples_cnt)

        return train_X, train_y, valid_X, valid_y, test_X, test_y

    def construct_model_dataset(self, remove_outliers, remove_non_training_attributes):
        """
        This is technically the wrong way of doing data split
        We are stacking all samples into a gaint matrix, and then doing the split
        Thus the training set contains A SUBSET OF the symbols, but each symbol has the
        FULL TIMELINE.

        Otherwise, the logic is more or less the same as construct_model_dataset_V2
        TODO: Should refactor...
        """

        X = []
        y = []

        for filename in os.listdir(self.data_source_dir):

            matrix = self.construct_data_matrix(
                filename, remove_non_training_attributes)

            n, d = matrix.shape

            self.feature_dimen = d

            # iterating each row for each stock ticker in the processed file
            for row_num in range(n):

                # if lag exceeds number of entries in the files, ignore
                if row_num + self.lag >= n:
                    break

                percentage_change_after_lag = matrix[row_num + self.lag][3]
                X.append(matrix[row_num: row_num + self.lag])
                y.append(self.convert_percentage_to_binary_label(
                    percentage_change_after_lag))

        if remove_outliers:
            print('Before removal', len(X))
            X, y = self.remove_outliers(X, y, self.outlier_threshold)
            print('After removal', len(X))

        train_X, train_y, valid_X, valid_y, test_X, test_y = self.construct_train_valid_test_set_from_X_y(
            X, y)

        negative_examples_cnt = np.count_nonzero(train_y == 0)
        positive_examples_cnt = np.count_nonzero(train_y == 1)
        print(positive_examples_cnt, negative_examples_cnt)

        # No need to oversample since threshold is balanced
        if self.threshold == 0.0:
            return train_X, train_y, valid_X, valid_y, test_X, test_y

        # augment negative examples when threshold is not 0 (thus classification is imbalanced)
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

        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]

        return train_X, train_y, valid_X, valid_y, test_X, test_y

    def convert_percentage_to_binary_label(self, percentage):
        if self.threshold >= 0.0:
            return 1 if percentage >= self.threshold else 0
        else:
            return 1 if percentage <= self.threshold else 0
