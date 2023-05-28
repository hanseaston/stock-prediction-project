
import os
import csv

import tensorflow as tf
from itertools import product
from tqdm import tqdm

from keras.losses import BinaryCrossentropy

from dataset_construction.binary_constructor import binary_constructor
from models.ModelSelector import select_model
from training.evaluator.BinaryEvaluator import BinaryEvaluator

######### Training configuration #########
BUFFER_SIZE = 128
NUM_EPOCH = 2

BATCH_SIZE = [1024 * 4, 1024 * 8]
LEARNING_RATE = [1e-3, 1e-2]
LAG = [10, 15]
LATENT_DIM = [32, 64]
THRESHOLD = [-0.01, 0.0, 0.01]
#######################################


def record_accuracy_results(model_directory, file_name, data, remove_dir=False):
    file_path = os.path.join(model_directory, file_name)
    if remove_dir:
        try:
            os.rmdir(model_directory)
        except:
            pass
        os.makedirs(model_directory, exist_ok=True)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[i+1, val]
                          for i, val in enumerate(data)])


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
        test_accuracy_hist = []

        for _ in tqdm(range(NUM_EPOCH)):

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
            score = evaluator.get_positive_accuracy_score()
            train_accuracy_hist.append(score)

            # for each epoch, record validation accuracy
            validation_predictions = model(valid_X, training=False)
            validation_loss = model.get_total_loss(
                valid_y, validation_predictions.numpy().flatten())
            evaluator = BinaryEvaluator(valid_y, validation_predictions, 0.5)
            score = evaluator.get_positive_accuracy_score()
            validation_accuracy_hist.append(score)

            # for each epoch, record test accuracy
            test_predictions = model(test_X, training=False)
            evaluator = BinaryEvaluator(test_y, test_predictions, 0.65)
            score = evaluator.get_positive_accuracy_score()
            test_accuracy_hist.append(score)

            train_loss_hist.append(
                train_loss.numpy() / len(batched_train_dataset))
            validation_loss_hist.append(validation_loss.numpy())

        model_directory = f"results/{batch_size}_{learning_rate}_{lag}_{latent_dim}_{threshold}"

        record_accuracy_results(
            model_directory, "train_accuracy.csv", train_accuracy_hist, True)
        record_accuracy_results(
            model_directory, "validation_accuracy.csv", validation_accuracy_hist)
        record_accuracy_results(
            model_directory, "test_accuracy.csv", test_accuracy_hist)
        record_accuracy_results(
            model_directory, "train_loss.csv", train_loss_hist)
        record_accuracy_results(
            model_directory, "validation_loss.csv", validation_loss_hist)

        # print('Saving model...')
        # tf.keras.models.save_model(model, f"{model_directory}/saved_model")
        # print('Done!')


if __name__ == '__main__':
    grid_search()
