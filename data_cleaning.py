import csv
import os
import matplotlib.pyplot as plt


def is_approximately_zero(value, tolerance=1e-5):
    try:
        float_value = float(value)
        return abs(float_value) < tolerance
    except ValueError:
        return False


if __name__ == '__main__':

    path = './dataset/polygon_processed'
    count_map = {}

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
            if row_count < 200:
                files_removed += 1
                os.remove(filepath)

            if zero_count not in count_map:
                count_map[zero_count] = []

            count_map[zero_count].append(filepath)
    print(f"removed {files_removed} files")

    length_map = {key: len(value) for key, value in count_map.items()}

    # keys = list(length_map.keys())
    # values = list(length_map.values())
    # plt.scatter(keys, values, marker='o')
    # plt.xlabel('Key')
    # plt.ylabel('Value')
    # plt.title('Map Data Plot')
    # plt.show()

    files_removed = 0
    for count, filenames in count_map.items():
        if count >= 200:
            for filename in filenames:
                os.remove(filename)
                files_removed += 1
    print(f"removed {files_removed} files")
