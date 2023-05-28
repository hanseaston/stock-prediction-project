import tensorflow as tf
from keras.layers import Dense, RNN, LSTMCell
from keras.models import Model
import numpy as np


######### Model configuration #########
LATENT_DIM = 32
L2_ALPHA = 0
#######################################


class LSTM(Model):

    def __init__(self, feature_dim, output_dim, loss_fn, **kwargs):

        super().__init__()

        self.l2_norm_alpha = L2_ALPHA

        self.encoding_layer = Dense(units=feature_dim,
                                    activation='tanh',
                                    kernel_initializer='glorot_uniform')

        self.rnn_layer = RNN(LSTMCell(
            units=LATENT_DIM,
            activation='tanh',
            kernel_initializer='glorot_uniform'),
            return_sequences=True)

        self.decoding_layer = Dense(units=output_dim,
                                    activation='sigmoid',
                                    kernel_initializer='glorot_uniform')

        self.loss_fn = loss_fn

    def call(self, args, training):

        input = args[0] if training else args

        encoded_feature = self.encoding_layer(input)

        rnn_output = self.rnn_layer(encoded_feature)

        predictions = self.decoding_layer(rnn_output[:, -1, :])

        return predictions

    def get_total_loss(self, correct_output, predicted_output):
        reg_loss = 0
        for train_var in self.trainable_variables:
            reg_loss += tf.nn.l2_loss(train_var)

        return self.loss_fn(correct_output, predicted_output) + reg_loss * self.l2_norm_alpha
