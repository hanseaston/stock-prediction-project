
import os

import tensorflow as tf
from tqdm import tqdm

from keras.losses import BinaryCrossentropy

from models.ModelSelector import select_model
from config import get_trained_model_path_from_version

from data.polygon.constructor.versions.V1 import PolygonV1Constructor

######### Dataset configuration #########
DATASET_NAME = 'sp500'
LAG = 10
THRESHOLD = 0.0
#########################################

######### Model configuration #########
MODEL_NAME = 'LSTM'
LATENT_DIM = 32
L2_ALPHA = 0
LOSS_FN = BinaryCrossentropy(from_logits=False)
#######################################

######### Training configuration #########
NUM_EPOCH = 10
BATCH_SIZE = 1024 * 8
LEARNING_RATE = 1e-2
OPTIMIZER = tf.keras.optimizers.legacy.Adam(
    learning_rate=LEARNING_RATE)
#######################################


def train_model():

    # construct the dataset with appropriate configuration
    dataset_constructor = PolygonV1Constructor(
        lag=LAG,
        threshold=THRESHOLD,
        date_start=None,
        date_end=None,
        dataset_name=DATASET_NAME)
    train_X, train_y, _, _, _, _ = dataset_constructor.build_dataset_all_tickers()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    batched_train_dataset = train_dataset.batch(BATCH_SIZE)
    feature_dim = dataset_constructor.get_feature_dimension()

    # initialize model
    args = {'feature_dim': feature_dim,
            'output_dim': 1,
            'loss_fn': LOSS_FN,
            'latent_dim': LATENT_DIM,
            'l2_alpha': L2_ALPHA}
    model = select_model(MODEL_NAME, args)

    # training loop
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

            OPTIMIZER.apply_gradients(
                zip(gradients, model.trainable_weights))

        print('Saving model...')
        model_dir = get_trained_model_path_from_version('v1', epoch)
        tf.keras.models.save_model(model, model_dir)
        print('Done!')


if __name__ == '__main__':
    train_model()
