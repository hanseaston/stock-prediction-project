import csv
import os
import matplotlib.pyplot as plt

from utils.utils import is_approximately_zero

if __name__ == '__main__':

    path = '../raw_data/polygon_processed_v2'
    zero_count_to_file_names = {}

    files_removed = 0

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        zero_count = 0
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                for value in row:
                    if is_approximately_zero(value):
                        zero_count += 1

            # if there are less than 200 entries, meaning it is a relatively
            # new stock, exclude from the dataset
            if row_count < 200:
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
        if count >= 200:
            for filename in filenames:
                os.remove(filename)
                files_removed += 1
    print(f"removed {files_removed} stocks due to too little movement")
