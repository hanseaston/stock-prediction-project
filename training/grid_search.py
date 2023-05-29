
import os
import csv

import tensorflow as tf
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.losses import BinaryCrossentropy

from dataset_construction.binary_constructor import binary_constructor
from models.ModelSelector import select_model
from training.evaluator.BinaryEvaluator import BinaryEvaluator

######### Training configuration #########
BUFFER_SIZE = 128
NUM_EPOCH = 100

BATCH_SIZE = [1024 * 4, 1024 * 8]
LEARNING_RATE = [1e-3]
LAG = [10, 15, 30]
LATENT_DIM = [32, 64, 128]
THRESHOLD = [-0.01, 0.0, 0.01]
#######################################


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


def grid_search():

    hyperparameter_combinations = product(
        BATCH_SIZE, LEARNING_RATE, LAG, LATENT_DIM, THRESHOLD)

    for batch_size, learning_rate, lag, latent_dim, threshold in hyperparameter_combinations:

        print(
            f"Training {batch_size} {learning_rate} {lag} {latent_dim} {threshold}...")

        # initialize dataset
        dataset_constructor = binary_constructor(lag=lag, threshold=threshold)
        train_X, train_y, valid_X, valid_y, test_X, test_y = dataset_constructor.construct_model_dataset()
        feature_dum = dataset_constructor.get_feature_dimension()
        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        batched_train_dataset = train_dataset.batch(batch_size)

        # initialize loss function
        loss_fn = BinaryCrossentropy(from_logits=False)

        # initialize model
        args = {'feature_dim': feature_dum,
                'output_dim': 1, 'loss_fn': loss_fn, 'latent_dim': latent_dim}
        model = select_model('AttnLSTM', args)
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate)

        train_accuracy_hist = []
        validation_accuracy_hist = []
        train_loss_hist = []
        validation_loss_hist = []
        test_accuracy_hist_50 = []
        test_accuracy_hist_60 = []
        test_accuracy_hist_70 = []

        for epoch in tqdm(range(NUM_EPOCH)):

            train_loss = 0

            for _, (x_batch_train, y_batch_train) in enumerate(batched_train_dataset):

                with tf.GradientTape() as tape:

                    predicted_output = model(
                        [x_batch_train, y_batch_train], training=True)

                    loss_value = model.get_total_loss(
                        y_batch_train, predicted_output)

                    train_loss += loss_value

                gradients = tape.gradient(
                    loss_value, model.trainable_variables)

                optimizer.apply_gradients(
                    zip(gradients, model.trainable_weights))

            # for each epoch, record train accuracy
            train_predictions = model(train_X, training=False)
            evaluator = BinaryEvaluator(train_y, train_predictions, 0.5)
            score, prediction_cnt = evaluator.get_positive_accuracy_score()
            train_accuracy_hist.append([epoch, score, prediction_cnt])

            # for each epoch, record validation accuracy
            validation_predictions = model(valid_X, training=False)
            validation_loss = model.get_total_loss(
                valid_y, validation_predictions.numpy().flatten())
            evaluator = BinaryEvaluator(valid_y, validation_predictions, 0.5)
            score, prediction_cnt = evaluator.get_positive_accuracy_score()
            validation_accuracy_hist.append([epoch, score, prediction_cnt])

            # for each epoch, record test accuracy with 0.5 threshold
            test_predictions = model(test_X, training=False)
            evaluator = BinaryEvaluator(test_y, test_predictions, 0.5)
            score, prediction_cnt = evaluator.get_positive_accuracy_score()
            test_accuracy_hist_50.append([epoch, score, prediction_cnt])

            # for each epoch, record test accuracy with 0.6 threshold
            test_predictions = model(test_X, training=False)
            evaluator = BinaryEvaluator(test_y, test_predictions, 0.6)
            score, prediction_cnt = evaluator.get_positive_accuracy_score()
            test_accuracy_hist_60.append([epoch, score, prediction_cnt])

            # for each epoch, record test accuracy with 0.7 threshold
            test_predictions = model(test_X, training=False)
            evaluator = BinaryEvaluator(test_y, test_predictions, 0.7)
            score, prediction_cnt = evaluator.get_positive_accuracy_score()
            test_accuracy_hist_70.append([epoch, score, prediction_cnt])

            # for each epoch, record lossed for both train and validation set
            train_loss_hist.append(
                [epoch, train_loss.numpy() / len(batched_train_dataset)])
            validation_loss_hist.append([epoch, validation_loss.numpy()])

        model_results_directory = f"results/{batch_size}_{learning_rate}_{lag}_{latent_dim}_{threshold}"

        record_results(
            model_results_directory, "train_accuracy.csv", train_accuracy_hist, ["epoch", "accuracy", "num_predictions"], True)
        record_results(
            model_results_directory, "validation_accuracy.csv", validation_accuracy_hist, ["epoch", "accuracy", "num_predictions"])
        record_results(
            model_results_directory, "test_accuracy.csv", test_accuracy_hist_50, ["epoch", "accuracy", "num_predictions"])
        record_results(
            model_results_directory, "test_accuracy.csv", test_accuracy_hist_60, ["epoch", "accuracy", "num_predictions"])
        record_results(
            model_results_directory, "test_accuracy.csv", test_accuracy_hist_70, ["epoch", "accuracy", "num_predictions"])
        record_results(
            model_results_directory, "train_loss.csv", train_loss_hist, ["epoch", "loss"])
        record_results(
            model_results_directory, "validation_loss.csv", validation_loss_hist, ["epoch", "loss"])

        _, ax = plt.subplots()

        file_path = os.path.join(model_results_directory, "accuracy.png")
        x = get_data(train_accuracy_hist, 0)  # epoch num
        train_y = get_data(train_accuracy_hist, 1)
        valid_y = get_data(validation_accuracy_hist, 1)
        test_y_50 = get_data(test_accuracy_hist_50, 1)
        test_y_60 = get_data(test_accuracy_hist_60, 1)
        test_y_70 = get_data(test_accuracy_hist_70, 1)
        ax.plot(x, train_y, label='train')
        ax.plot(x, valid_y, label='validation')
        ax.plot(x, test_y_50, label="test 0.5")
        ax.plot(x, test_y_60, label="test 0.6")
        ax.plot(x, test_y_70, label="test 0.7")
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.legend()
        plt.savefig(file_path)

        plt.clf()  # clean the plot for the next result
        _, ax = plt.subplots()

        file_path = os.path.join(model_results_directory, "loss.png")
        x = get_data(train_loss_hist, 0)  # epoch num
        train_y = get_data(train_loss_hist, 1)
        valid_y = get_data(validation_loss_hist, 1)
        ax.plot(x, train_y, label='train')
        ax.plot(x, valid_y, label='validation')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        plt.savefig(file_path)

        # print('Saving model...')
        # tf.keras.models.save_model(model, f"{model_directory}/saved_model")
        # print('Done!')


if __name__ == '__main__':
    grid_search()
