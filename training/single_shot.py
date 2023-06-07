
import os

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from keras.losses import BinaryCrossentropy

from dataset_construction.binary_constructor import binary_constructor
from models.ModelSelector import select_model
from training.evaluator.BinaryEvaluator import BinaryEvaluator
from utils.utils import get_data, record_results

######### Training configuration #########
BUFFER_SIZE = 128
NUM_EPOCH = 20

LAG = 10
BATCH_SIZE = 1024 * 8
LEARNING_RATE = 1e-2
LATENT_DIM = 32
L2_ALPHA = 0
#######################################


def train_single_model(dataset_name, model_name, threshold, data_path):

    print(f'training {model_name} model...')

    dataset_constructor = binary_constructor(
        lag=LAG, threshold=threshold, data_path=data_path)
    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset_constructor.construct_model_dataset(
        False, True)
    feature_dim = dataset_constructor.get_feature_dimension()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    batched_train_dataset = train_dataset.batch(BATCH_SIZE)

    # initialize loss function
    loss_fn = BinaryCrossentropy(from_logits=False)

    # initialize model
    args = {'feature_dim': feature_dim,
            'output_dim': 1, 'loss_fn': loss_fn, 'latent_dim': LATENT_DIM, 'l2_alpha': L2_ALPHA}
    model = select_model('AttnLSTM', args)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE)

    train_accuracy_hist = []
    validation_accuracy_hist = []
    train_loss_hist = []
    validation_loss_hist = []
    test_accuracy_hist_50 = []
    test_accuracy_hist_60 = []
    test_accuracy_hist_70 = []
    test_accuracy_hist_80 = []
    test_accuracy_hist_90 = []

    for epoch in tqdm(range(NUM_EPOCH)):

        train_loss = 0

        for _, (x_batch_train, y_batch_train) in enumerate(batched_train_dataset):

            with tf.GradientTape() as tape:

                predicted_output = model(
                    x_batch_train, training=True)

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

        # for each epoch, record loss for both train and validation set
        train_loss_hist.append(
            [epoch, train_loss.numpy() / len(batched_train_dataset)])
        validation_loss_hist.append([epoch, validation_loss.numpy()])

        # record test for threshold 0.5
        test_predictions = model(test_X, training=False)
        evaluator = BinaryEvaluator(test_y, test_predictions, 0.5)
        score, prediction_cnt = evaluator.get_positive_accuracy_score()
        test_accuracy_hist_50.append([epoch, score, prediction_cnt])

        # record test for threshold 0.6
        evaluator = BinaryEvaluator(test_y, test_predictions, 0.6)
        score, prediction_cnt = evaluator.get_positive_accuracy_score()
        test_accuracy_hist_60.append([epoch, score, prediction_cnt])

        # record test for threshold 0.7
        evaluator = BinaryEvaluator(test_y, test_predictions, 0.7)
        score, prediction_cnt = evaluator.get_positive_accuracy_score()
        test_accuracy_hist_70.append([epoch, score, prediction_cnt])

        # record test for threshold 0.8
        evaluator = BinaryEvaluator(test_y, test_predictions, 0.8)
        score, prediction_cnt = evaluator.get_positive_accuracy_score()
        test_accuracy_hist_80.append([epoch, score, prediction_cnt])

        # record test for threshold 0.9
        evaluator = BinaryEvaluator(test_y, test_predictions, 0.9)
        score, prediction_cnt = evaluator.get_positive_accuracy_score()
        test_accuracy_hist_90.append([epoch, score, prediction_cnt])

    model_results_directory = f"results/{dataset_name}/{model_name}"

    record_results(
        model_results_directory, "train_accuracy.csv", train_accuracy_hist, ["epoch", "accuracy", "num_predictions"], True)
    record_results(
        model_results_directory, "validation_accuracy.csv", validation_accuracy_hist, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_50.csv", test_accuracy_hist_50, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_60.csv", test_accuracy_hist_60, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_70.csv", test_accuracy_hist_70, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_80.csv", test_accuracy_hist_80, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_90.csv", test_accuracy_hist_90, ["epoch", "accuracy", "num_predictions"])

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
    test_y_80 = get_data(test_accuracy_hist_80, 1)
    test_y_90 = get_data(test_accuracy_hist_90, 1)

    ax.plot(x, train_y, label='train')
    ax.plot(x, valid_y, label='validation')
    ax.plot(x, test_y_50, label='test_50')
    ax.plot(x, test_y_60, label='test_60')
    ax.plot(x, test_y_70, label='test_70')
    ax.plot(x, test_y_80, label='test_80')
    ax.plot(x, test_y_90, label='test_90')
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

    model_dir = os.path.join(model_results_directory, "trained_model")

    print('Saving model...')
    tf.keras.models.save_model(model, model_dir)
    print('Done!')

    return model


def evaluate_hybrid_model(neutral_threshold_model, evaluation_dataset_path):

    dataset_constructor = binary_constructor(
        lag=LAG, threshold=0.0, data_path=evaluation_dataset_path)

    test_X, test_y = dataset_constructor.construct_evaluation_dataset()

    neutral_classifier_predictions = neutral_threshold_model(
        test_X, training=False)
    neutral_classifier_predictions = (
        np.array(neutral_classifier_predictions) > 0.9).astype(int)

    hybrid_predictions = []
    for i in range(len(neutral_classifier_predictions)):
        neu = neutral_classifier_predictions[i]
        if neu == 1:
            hybrid_predictions.append(1)
        else:
            hybrid_predictions.append(0)

    evaluator = BinaryEvaluator(test_y, hybrid_predictions, 493)
    score, prediction_cnt = evaluator.get_positive_accuracy_score()
    print(f"Final accurcacy is {score} out of {prediction_cnt} predictions")


if __name__ == '__main__':

    TRAIN_MODE = True
    EVALUATION_MODE = True

    dataset_name = 'sp500'

    if TRAIN_MODE:

        data_path = '../raw_data/sp500_2014_2023_processed'
        train_single_model(dataset_name, "neutral", 0, data_path)

    if EVALUATION_MODE:
        neutral_threshold_model_path = f"results/{dataset_name}/neutral/trained_model"
        neutral_threshold_model = tf.keras.models.load_model(
            neutral_threshold_model_path)

        ### TODO: change this when necessary ###
        evaluation_dataset_path = '../raw_data/sp500_2021_2023_processed'

        evaluate_hybrid_model(neutral_threshold_model, evaluation_dataset_path)
