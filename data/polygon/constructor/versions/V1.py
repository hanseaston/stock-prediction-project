import os
import numpy as np

from data.polygon.constructor.BaseContructor import PolygonBaseConstructor
from config import get_ml_dataset_path_from_name


class PolygonV1Constructor(PolygonBaseConstructor):
    """
    !!! Version 1 of the model data constructor !!!
    Binary classification with flexible dataset choice, time range, and threshold range.
    """

    def __init__(self, lag, threshold, date_start, date_end, dataset_name):
        super().__init__(dataset_name)

        # User-defined variables
        self.lag = lag
        self.classification_threshold = threshold
        self.date_start = date_start
        self.date_end = date_end
        self.dataset_name = dataset_name

        # Version-defined variables
        self.training_ratio = 0.8
        self.validation_ratio = 0.1
        self.columns = ['L1_CP_R', 'CP_HP_R', 'CP_LP_R', 'L5_CP_SMA_R', 'L10_CP_SMA_R', 'L20_CP_SMA_R',
                        'L30_CP_SMA_R', 'L5_TV_SMA_R', 'L10_TV_SMA_R', 'L20_TV_SMA_R', 'L30_TV_SMA_R']
        self.skip_rows = 30  # since the 30-SMA is missing in the first 30 rows
        self.feature_dimension = 11  # same as length of the columns

        # Train, validation, and test dataset
        self.train_X = []
        self.train_y = []
        self.valid_X = []
        self.valid_y = []
        self.test_X = []
        self.test_y = []

    def build_dataset_all_tickers(self):
        """
        Builds the train, validation, and test dataset for all processed ticker data.
        """

        for file_name in os.listdir(self.processed_dataset_path):

            # selects the appropriate subsection of the ticker dataframe
            df = self.selector.select_data(
                file_name=file_name,
                date_start=self.date_start,
                date_end=self.date_end,
                skip_rows=self.skip_rows,
                column_names=self.columns)

            self.build_dataset_single_ticker(df)

        return self.shuffle_dataset()

    def build_dataset_single_ticker(self, df):
        """
        Builds the train, validation, and test dataset for individual processed ticker.
        """

        n, _ = df.shape

        # iterating each row for each stock ticker in the processed file
        for row_num in range(n):

            # if lag exceeds number of entries in the files, ignore
            if row_num + self.lag >= n:
                break

            # get percentage change on the i+1 day
            percentage_change_after_lag = df.iloc[row_num +
                                                  self.lag][df.columns.get_loc('L1_CP_R')]

            # get the X and y pair
            X = df.loc[row_num: row_num + self.lag - 1].to_numpy()
            y = self.convert_percentage_to_binary_label(
                percentage_change_after_lag)

            # add the (X, y) pair to the dataset, depending on whether it is train, valid, or test
            self.add_to_dataset(current_row=row_num, total_row=n, X=X, y=y)

    def add_to_dataset(self, current_row, total_row, X, y):
        """
        Based on current row, determine whether to add the (X, y) pair to 
        train, validation, or test dataset.
        """

        train_index = int(total_row * self.training_ratio)
        validation_index = int(
            total_row * (self.training_ratio + self.validation_ratio))

        if current_row <= train_index:
            self.train_X.append(X)
            self.train_y.append(y)
        elif current_row <= validation_index:
            self.valid_X.append(X)
            self.valid_y.append(y)
        else:
            self.test_X.append(X)
            self.test_y.append(y)

    def convert_percentage_to_binary_label(self, percentage):
        """
        Conditions on a specific attribute of the dataframe (e.x closing price),
        and uses the classification threshold to assign the label y. 
        """

        if self.classification_threshold == 0.0:
            return 1 if percentage >= self.classification_threshold else 0

    def get_feature_dimension(self):
        """
        Returns the number of features in the dataset
        """

        return self.feature_dimension

    def shuffle_dataset(self):
        """
        Converts the dataset to numpy form, and shuffles the train dataset.
        """

        train_X = np.array(self.train_X)
        train_y = np.array(self.train_y)
        valid_X = np.array(self.valid_X)
        valid_y = np.array(self.valid_y)
        test_X = np.array(self.test_X)
        test_y = np.array(self.test_y)
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]
        return train_X, train_y, valid_X, valid_y, test_X, test_y

    def pickle_save_dataset(self):
        dataset_dir = os.path.join(
            get_ml_dataset_path_from_name(self.dataset_name), 'v1')
        self.train_X_dataset_dir = os.path.join(dataset_dir, 'training_X.np')
        self.train_y_dataset_dir = os.path.join(dataset_dir, 'training_y.np')
        self.valid_X_dataset_dir = os.path.join(dataset_dir, 'validation_X.np')
        self.valid_y_dataset_dir = os.path.join(dataset_dir, 'validation_y.np')
        self.test_X_dataset_dir = os.path.join(dataset_dir, 'test_X.np')
        self.test_y_dataset_dir = os.path.join(dataset_dir, 'test_y.np')
        np.savetxt(self.train_X_dataset_dir, np.array(self.train_X))
        np.savetxt(self.train_y_dataset_dir, np.array(self.train_y))
        np.savetxt(self.valid_X_dataset_dir, np.array(self.valid_X))
        np.savetxt(self.valid_y_dataset_dir, np.array(self.valid_y))
        np.savetxt(self.test_X_dataset_dir, np.array(self.test_X))
        np.savetxt(self.test_y_dataset_dir, np.array(self.test_y))
