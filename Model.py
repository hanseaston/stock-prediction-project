import tensorflow as tf
from keras.initializers import GlorotUniform, Zeros
from keras.layers import Dense, RNN, LSTMCell
from keras.models import Model
from tensorflow import tanh, tensordot, reduce_sum, concat


class AdvAttentionLSTM(Model):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.attn_lstm_layer = AttnLSTM(feature_dim, latent_dim)

    def call(self, input):
        return self.attn_lstm_layer(input)


class AttnLSTM(Model):

    def __init__(self, feature_dim, latent_dim):

        super().__init__()

        self.encoding_layer = Dense(units=feature_dim,
                                    activation='tanh',
                                    kernel_initializer='glorot_uniform')

        self.rnn_layer = RNN(LSTMCell(
            units=latent_dim,
            activation='tanh',
            kernel_initializer='glorot_uniform'),
            return_sequences=True)

        self.attention_weight = tf.Variable(
            initial_value=GlorotUniform()((latent_dim, latent_dim)),
            dtype=tf.float32,
            name='attention_weight')

        self.attention_bias = tf.Variable(
            initial_value=Zeros()(latent_dim),
            dtype=tf.float32,
            name='attention_bias')

        # TODO: think of a better name
        self.attention_u = tf.Variable(
            initial_value=Zeros()(latent_dim),
            dtype=tf.float32,
            name='attention_u')

        self.decoding_layer = Dense(units=1,
                                    activation=None,
                                    kernel_initializer='glorot_uniform')

    def call(self, input):

        encoded_feature = self.encoding_layer(input)

        rnn_output = self.rnn_layer(encoded_feature)

        attention_latent = tanh(
            tensordot(rnn_output, self.attention_weight, axes=1) + self.attention_bias)

        attention_score = tensordot(attention_latent, self.attention_u, axes=1)

        attention_score = tf.nn.softmax(attention_score)

        attention_output = reduce_sum(
            rnn_output * tf.expand_dims(attention_score, -1), 1)

        output = concat(
            [rnn_output[:, -1, :], attention_output], axis=1)

        predictions = self.decoding_layer(output)

        return predictions
