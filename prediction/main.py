from datetime import datetime
import numpy as np
import tensorflow as tf
import os

from data_pipeline.fetch_polygon import PolygonParser
from data_pipeline.data_preprocess import preprocess_all_tickers
from dataset_construction.binary_constructor import binary_constructor
from utils.utils import get_file_name


raw_data_path = '../prediction/data'
processed_data_path = '../prediction/processed'
neutral_threshold_model_path = "../training/results/hybrid/neutral/trained_model"
positive_threshold_model_path = "../training/results/hybrid/positive/trained_model"
negative_threshold_model_path = "../training/results/hybrid/negative/trained_model"
lag = 10


def fetch_data():
    parser = PolygonParser(raw_data_path)
    date_start = "2023-04-01"  # needs more efficien timplementation
    date_end = datetime.now().strftime("%Y-%m-%d")
    parser.parse_sp500_tickers(date_start, date_end)


def preprocess_data():
    preprocess_all_tickers(raw_data_path, processed_data_path)


def make_predictions_for_tomorrow():
    # get the data
    dataset_constructor = binary_constructor(lag, np.NAN, processed_data_path)
    data = dataset_constructor.construct_prediction_dataset()

    # load the model
    neutral_threshold_model = tf.keras.models.load_model(
        neutral_threshold_model_path)
    positive_threshold_model = tf.keras.models.load_model(
        positive_threshold_model_path)
    negative_threshold_model = tf.keras.models.load_model(
        negative_threshold_model_path)

    neutral_classifier_predictions = neutral_threshold_model(
        data, training=False)
    neutral_classifier_predictions = (
        np.array(neutral_classifier_predictions) > 0.7).astype(int)
    positive_classifier_predictions = positive_threshold_model(
        data, training=False)
    positive_classifier_predictions = (
        np.array(positive_classifier_predictions) > 0.8).astype(int)
    negative_classifier_predictions = negative_threshold_model(
        data, training=False)
    negative_classifier_predictions = (
        np.array(negative_classifier_predictions) > 0.8).astype(int)

    hybrid_predictions = []
    positive_prediction_idx = []

    for i in range(len(neutral_classifier_predictions)):
        neu = neutral_classifier_predictions[i]
        pos = positive_classifier_predictions[i]
        neg = negative_classifier_predictions[i]
        if neu == 1 and pos == 1 and neg == 0:
            hybrid_predictions.append(1)
            positive_prediction_idx.append(i)
        else:
            hybrid_predictions.append(0)
    hybrid_predictions = np.array(hybrid_predictions)

    print(f"Made {np.count_nonzero(hybrid_predictions == 1)} predictions")

    ticker_symbols = []
    for ticker_file in os.listdir(processed_data_path):
        ticker_symbols.append(get_file_name(ticker_file))
    print(
        f"The stocks you should buy are: {np.array(ticker_symbols)[positive_prediction_idx]}")


if __name__ == '__main__':
    # fetch_data()
    # preprocess_data()
    make_predictions_for_tomorrow()
