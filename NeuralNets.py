from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
import tensorflow as tf


class NeuralNets(Model):

    def __init__(self, feature_dim, sequence_len, l2_norm_alpha, **kwargs):

        super().__init__()

        tf.random.set_seed(123456)

        self.l2_norm_alpha = l2_norm_alpha

        # Define sequential model
        self.model = Sequential()

        # Flatten input first to match with Dense layer input shape
        self.model.add(Flatten(input_shape=(sequence_len, feature_dim)))

        # First Dense layer
        self.model.add(Dense(sequence_len, activation='relu'))

        # Second Dense layer
        self.model.add(Dense(feature_dim, activation='relu'))

        # Output layer for binary classification
        self.model.add(Dense(1, activation=None))

        self.loss_fn = tf.losses.Hinge()

    def call(self, args, training):
        input = args[0] if training else args
        return self.model(input)

    def get_total_loss(self, correct_output, predicted_output):
        reg_loss = 0
        for train_var in self.trainable_variables:
            reg_loss += tf.nn.l2_loss(train_var)
        return self.loss_fn(correct_output, predicted_output) + reg_loss * self.l2_norm_alpha
