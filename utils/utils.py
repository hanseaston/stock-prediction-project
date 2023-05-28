import os
import tensorflow as tf


def is_approximately_zero(value, tolerance=1e-5):
    try:
        float_value = float(value)
        return abs(float_value) < tolerance
    except ValueError:
        return False


def remove_all_files_from_dir(dir_path):
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
