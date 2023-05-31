
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
NUM_EPOCH = 25

LAG = 10
BATCH_SIZE = 1024 * 16
LEARNING_RATE = 1e-2
LATENT_DIM = 64
L2_ALPHA = 1e-5
#######################################


def train_single_model(model_name, threshold, data_path):

    print(f'training {model_name} model...')

    dataset_constructor = binary_constructor(
        lag=LAG, threshold=threshold, data_path=data_path)
    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset_constructor.construct_model_dataset()
    feature_dim = dataset_constructor.get_feature_dimension()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    batched_train_dataset = train_dataset.batch(BATCH_SIZE)

    # initialize loss function
    loss_fn = BinaryCrossentropy(from_logits=False)

    # initialize model
    args = {'feature_dim': feature_dim,
            'output_dim': 1, 'loss_fn': loss_fn, 'latent_dim': LATENT_DIM, 'l2_alpha': L2_ALPHA}
    model = select_model('LSTM', args)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE)

    train_accuracy_hist = []
    validation_accuracy_hist = []
    train_loss_hist = []
    validation_loss_hist = []
    test_accuracy_hist = []

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
        f1_score = evaluator.get_f1_score()
        train_accuracy_hist.append([epoch, score, prediction_cnt])

        # for each epoch, record validation accuracy
        validation_predictions = model(valid_X, training=False)
        validation_loss = model.get_total_loss(
            valid_y, validation_predictions.numpy().flatten())
        evaluator = BinaryEvaluator(valid_y, validation_predictions, 0.5)
        score, prediction_cnt = evaluator.get_positive_accuracy_score()
        validation_accuracy_hist.append([epoch, score, prediction_cnt])

        # for each epoch, record lossed for both train and validation set
        train_loss_hist.append(
            [epoch, train_loss.numpy() / len(batched_train_dataset)])
        validation_loss_hist.append([epoch, validation_loss.numpy()])

        test_predictions = model(test_X, training=False)
        evaluator = BinaryEvaluator(test_y, test_predictions, 0.8)
        score, prediction_cnt = evaluator.get_positive_accuracy_score()
        test_accuracy_hist.append([epoch, score, prediction_cnt])

    model_results_directory = f"results/hybrid/{model_name}"

    record_results(
        model_results_directory, "train_accuracy.csv", train_accuracy_hist, ["epoch", "accuracy", "num_predictions"], True)
    record_results(
        model_results_directory, "validation_accuracy.csv", validation_accuracy_hist, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy.csv", test_accuracy_hist, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "train_loss.csv", train_loss_hist, ["epoch", "loss"])
    record_results(
        model_results_directory, "validation_loss.csv", validation_loss_hist, ["epoch", "loss"])

    _, ax = plt.subplots()

    file_path = os.path.join(model_results_directory, "accuracy.png")
    x = get_data(train_accuracy_hist, 0)  # epoch num
    train_y = get_data(train_accuracy_hist, 1)
    valid_y = get_data(validation_accuracy_hist, 1)
    test_y = get_data(test_accuracy_hist, 1)

    ax.plot(x, train_y, label='train')
    ax.plot(x, valid_y, label='validation')
    ax.plot(x, test_y, label='test')
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


def evaluate_hybrid_model(neutral_threshold_model, positive_threshold_model, negative_threshold_model):

    dataset_constructor = binary_constructor(
        lag=LAG, threshold=0.0, data_path="../raw_data/polygon_processed/")

    test_X, test_y = dataset_constructor.construct_evaluation_dataset()

    # dataset_constructor = binary_constructor(
    #     lag=LAG, threshold=0.0, data_path="../raw_data/polygon_processed_v2/")

    # train_X, train_y, valid_X, valid_y, test_X, test_y = dataset_constructor.construct_model_dataset()

    neutral_classifier_predictions = neutral_threshold_model(
        test_X, training=False)
    neutral_classifier_predictions = (
        np.array(neutral_classifier_predictions) > 0.8).astype(int)
    positive_classifier_predictions = positive_threshold_model(
        test_X, training=False)
    positive_classifier_predictions = (
        np.array(positive_classifier_predictions) > 0.9).astype(int)
    negative_classifier_predictions = negative_threshold_model(
        test_X, training=False)
    negative_classifier_predictions = (
        np.array(negative_classifier_predictions) > 0.9).astype(int)

    hybrid_predictions = []
    for i in range(len(neutral_classifier_predictions)):
        neu = neutral_classifier_predictions[i]
        pos = positive_classifier_predictions[i]
        neg = negative_classifier_predictions[i]
        if neu == 1 and pos == 1 and neg == 0:
            hybrid_predictions.append(1)
        else:
            hybrid_predictions.append(0)

    evaluator = BinaryEvaluator(test_y, hybrid_predictions, 493)
    score, prediction_cnt = evaluator.get_positive_accuracy_score()
    print(f"Final accurcacy is {score} out of {prediction_cnt} predictions")


if __name__ == '__main__':

    TRAIN_MODE = True
    EVALUATION_MODE = True

    if TRAIN_MODE:
        model_data_path = "../raw_data/polygon_processed_v2/"
        # training model
        train_single_model("neutral", 0.0, model_data_path)
        train_single_model("positive", 0.01, model_data_path)
        train_single_model("negative", -0.01, model_data_path)

    if EVALUATION_MODE:
        neutral_threshold_model_path = "results/hybrid/neutral/trained_model"
        neutral_threshold_model = tf.keras.models.load_model(
            neutral_threshold_model_path)
        positive_threshold_model_path = "results/hybrid/positive/trained_model"
        positive_threshold_model = tf.keras.models.load_model(
            positive_threshold_model_path)
        negative_threshold_model_path = "results/hybrid/negative/trained_model"
        negative_threshold_model = tf.keras.models.load_model(
            negative_threshold_model_path)
        evaluate_hybrid_model(neutral_threshold_model,
                              positive_threshold_model, negative_threshold_model)
