import tensorflow as tf
from keras.layers import Dense, RNN, LSTMCell
from keras.models import Model


class LatentLSTM(Model):

    def __init__(self, feature_dim, output_dim, loss_fn, latent_dim, l2_alpha, **kwargs):

        super().__init__()

        self.encoding_layer = Dense(units=feature_dim,
                                    activation='tanh',
                                    kernel_initializer='glorot_uniform')

        self.rnn_layer = RNN(LSTMCell(
            units=latent_dim,
            activation='tanh',
            kernel_initializer='glorot_uniform'),
            return_sequences=True)

        self.decoding_layer = Dense(units=output_dim,
                                    activation='sigmoid',
                                    kernel_initializer='glorot_uniform')

        self.loss_fn = loss_fn
        self.l2_alpha = l2_alpha

    def call(self, input):

        encoded_feature = self.encoding_layer(input)

        rnn_output = self.rnn_layer(encoded_feature)

        predictions = self.decoding_layer(rnn_output[:, -1, :])

        return predictions

    def get_total_loss(self, correct_output, predicted_output):
        reg_loss = 0
        for train_var in self.trainable_variables:
            reg_loss += tf.nn.l2_loss(train_var)
        return self.loss_fn(correct_output, predicted_output) + reg_loss * self.l2_alpha

    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_dim': self.encoding_layer.units,
            'output_dim': self.decoding_layer.units,
            'loss_fn': tf.keras.losses.serialize(self.loss_fn),
            'latent_dim': self.rnn_layer.cell.units,
            'l2_alpha': self.l2_alpha,
            'encoding_layer_config': self.encoding_layer.get_config(),
            'decoding_layer_config': self.decoding_layer.get_config(),
            'rnn_layer_config': self.rnn_layer.get_config()
        })
        return config
