import tensorflow as tf
from keras.layers import Dense, RNN, LSTMCell
from keras.models import Model


class LSTM(Model):

    def __init__(self, feature_dim, latent_dim, l2_norm_alpha, **kwargs):

        super().__init__()
        tf.random.set_seed(123456)

        self.l2_norm_alpha = l2_norm_alpha

        self.encoding_layer = Dense(units=feature_dim,
                                    activation='tanh',
                                    kernel_initializer='glorot_uniform')

        self.rnn_layer = RNN(LSTMCell(
            units=latent_dim,
            activation='tanh',
            kernel_initializer='glorot_uniform'),
            return_sequences=True)

        self.decoding_layer = Dense(units=1,
                                    activation=None,
                                    kernel_initializer='glorot_uniform')

        self.loss_fn = tf.losses.Hinge()

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
