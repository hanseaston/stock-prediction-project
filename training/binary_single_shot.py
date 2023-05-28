import tensorflow as tf
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_construction.binary_constructor import binary_constructor
from models.ModelSelector import select_model
from training.evaluator.BinaryEvaluator import BinaryEvaluator

from keras.losses import BinaryCrossentropy

######### Training configuration #########
BATCH_SIZE = 1024 * 4
BUFFER_SIZE = 128
NUM_EPOCH = 100
LEARNING_RATE = 1e-3
LAG = 10
#######################################


def single_shot():

    # initialize dataset
    dataset_constructor = binary_constructor(LAG)
    train_X, train_y, valid_X, valid_y, test_X, test_y = dataset_constructor.construct_model_dataset()
    feature_dum = dataset_constructor.get_feature_dimension()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    batched_train_dataset = train_dataset.batch(BATCH_SIZE)

    # initialize loss function
    loss_fn = BinaryCrossentropy(from_logits=False)

    # initialize model
    args = {'feature_dim': feature_dum, 'output_dim': 1, 'loss_fn': loss_fn}
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

        # for each epoch, report train accuracy
        train_predictions = model(train_X, training=False)

        evaluator = BinaryEvaluator(train_y, train_predictions, 0.5)

        evaluator.positive_accuracy_score("train")
        evaluator.negative_accuracy_score("train")
        evaluator.accuracy_score('train')
        print()

        # for each epoch, report test accuracy
        test_predictions = model(test_X, training=False)

        evaluator = BinaryEvaluator(
            test_y, test_predictions, 0.75)

        evaluator.positive_accuracy_score("test")
        evaluator.negative_accuracy_score("test")
        evaluator.accuracy_score('test')

        print(f"loss: {epoch_loss}")
        print("----------------------------------")

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
