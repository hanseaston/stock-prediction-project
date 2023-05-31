import os
import tensorflow as tf
import numpy as np
import csv


def record_results(model_directory, file_name, data, header, remove_dir=False):
    file_path = os.path.join(model_directory, file_name)
    if remove_dir:
        try:
            os.rmdir(model_directory)
        except:
            pass
        os.makedirs(model_directory, exist_ok=True)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


def get_data(data_arr, index):
    y = []
    for data in data_arr:
        y.append(data[index])
    return y


def is_approximately_zero(value, tolerance=1e-5):
    try:
        float_value = float(value)
        return abs(float_value) < tolerance
    except ValueError:
        return False


def remove_all_files_from_dir(dir_path):
    if not os.path.exists(dir_path):
        return
    files_removed = 0
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            files_removed += 1
    print(f"Done removing ... {files_removed} files")


def check_all_zeros(lst):
    for element in lst:
        if element != 0:
            return False
    return True


def check_tensor_equal(tensor1, tensor2):
    return tf.reduce_all(tf.equal(tensor1, tensor2))


def remove_rows_from_matrix(X, arr):
    mask = np.ones(X.shape[0], dtype=bool)  # Create a boolean mask
    mask[arr] = False  # Set indices specified in arr to False
    return X[mask]  # Apply the mask to remove the corresponding rows


def get_file_name(file):
    return file.split('.')[0]
