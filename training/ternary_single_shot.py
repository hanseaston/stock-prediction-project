import tensorflow as tf
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from dataset_construction.ternary_constructor import ternary_constructor
from models.ModelSelector import select_model
from losses.WeightedCrossEntropy import WeightedCrossEntropy

######### Training configuration #########
BATCH_SIZE = 1024
NUM_EPOCH = 50
LEARNING_RATE = 1e-3
LAG = 10
TREND_SCALARS = tf.constant([1.0, 1.0, 1.0])
#######################################


def single_shot():

    # initialize dataset
    dataset_constructor = ternary_constructor(LAG)
    train_X, train_y, valid_X, valid_y, test_X, test_Y = dataset_constructor.construct_model_dataset()
    feature_dum = dataset_constructor.get_feature_dimension()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    batched_train_dataset = train_dataset.batch(BATCH_SIZE)

    # initialize ratio weights
    trend_ratios = dataset_constructor.get_trend_ratios()
    trend_weights = TREND_SCALARS

    # initialize loss function
    loss_fn = WeightedCrossEntropy(trend_weights)

    # initialize model
    args = {'feature_dim': feature_dum, 'output_dim': 3, 'loss_fn': loss_fn}
    model = select_model('LSTM', args)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)

    train_accuracy_hist = []
    test_accuracy_hist = []

    for _ in tqdm(range(NUM_EPOCH)):

        epoch_loss = 0

        for _, (x_batch_train, y_batch_train) in enumerate(batched_train_dataset):

            with tf.GradientTape() as tape:

                predicted_output = model(
                    [x_batch_train, y_batch_train], training=True)
                loss_value = model.get_total_loss(
                    y_batch_train, predicted_output)
                epoch_loss += loss_value

            gradients = tape.gradient(
                loss_value, model.trainable_variables)

            optimizer.apply_gradients(
                zip(gradients, model.trainable_weights))

        # for each epoch, report train accuracyc
        train_predictions = model(train_X, training=False)
        train_predictions = (np.sign(train_predictions) + 1) / 2
        accuracy = accuracy_score(train_predictions, train_y)
        accuracy = np.round(accuracy * 100, 2)
        print(
            f"train accuracy with current model: {accuracy}%")
        train_accuracy_hist.append(accuracy)

        # for each epoch, report test accuracy
        test_predictions = model(test_X, training=False)
        test_predictions = (np.sign(test_predictions) + 1) / 2
        accuracy = accuracy_score(test_predictions, test_Y)
        accuracy = np.round(accuracy * 100, 2)
        print(
            f"test accuracy with current model: {accuracy}%")

        print(f"loss: {epoch_loss}")
        test_accuracy_hist.append(accuracy)

    print(len(train_accuracy_hist))

    # plots
    epochs = range(1, len(train_accuracy_hist) + 1)
    plt.plot(epochs, train_accuracy_hist, 'bo-', label='Train Accuracy')
    plt.plot(epochs, test_accuracy_hist, 'go-', label='Test Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    single_shot()
