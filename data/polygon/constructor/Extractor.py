def extract_data_between_range(self):
        raw_dataset_path = self.get_raw_dataset_path_from_name(
            self.dataset_name)
        # if both dates are missing, we assume the user wants all data
        if self.date_start is None and self.date_end is None:
            return pd.read_csv(raw_dataset_path)

    def get_raw_dataset_path_from_name(dataset_name):
        if dataset_name == 'sp500':
            return PATHS['polygon_dataset_sp500_raw']
        elif dataset_name == 'nasdaq':
            return PATHS['polygon_dataset_nasdaq_raw']