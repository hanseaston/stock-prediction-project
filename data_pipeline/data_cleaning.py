import csv
import os
import matplotlib.pyplot as plt

from utils.utils import is_approximately_zero

data_path = '../raw_data/sp500_2014_2019_processed'


"""
Doing some very simple data cleaning... 
First, remove files that do not have enough entries
Secondly, if the stock is very inactive (e.x a lot of the percentage change is around
0%, then remove such files as well).
The thresholds to determine these conditions are parameters (see main method at the bottom)
"""


def clean_data(minimum_num_entries, maxmimum_zero_movements):
    zero_count_to_file_names = {}

    files_removed = 0

    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        zero_count = 0
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                for value in row:
                    if is_approximately_zero(value):
                        zero_count += 1

            # if there are less than minimum_num_entries entries, meaning it is a relatively
            # new stock, exclude from the dataset
            if row_count < minimum_num_entries:
                files_removed += 1
                os.remove(filepath)

            if zero_count not in zero_count_to_file_names:
                zero_count_to_file_names[zero_count] = []

            zero_count_to_file_names[zero_count].append(filepath)
    print(f"Removed {files_removed} stocks due to not enough entries")

    zero_count_to_files_count = {key: len(value)
                                 for key, value in zero_count_to_file_names.items()}

    # graph on relationship between zero counts and file counts
    keys = list(zero_count_to_files_count.keys())
    values = list(zero_count_to_files_count.values())
    plt.scatter(keys, values, marker='o')
    plt.xlabel('# of zero entries for a ticker')
    plt.ylabel('# of files')
    plt.show()

    files_removed = 0
    for count, filenames in zero_count_to_file_names.items():
        # if there's a lot of zero entries, this means the prive movement is minimal
        # thus not likely a good candidate for the dataset
        if count >= maxmimum_zero_movements:
            for filename in filenames:
                os.remove(filename)
                files_removed += 1
    print(f"removed {files_removed} stocks due to too little movement")


if __name__ == '__main__':
    clean_data(500, 500)
