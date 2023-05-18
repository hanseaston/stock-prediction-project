import tensorflow as tf
from keras.initializers import GlorotUniform, Zeros
from keras.layers import Dense, RNN, LSTMCell
from keras.models import Model
from tensorflow import tanh, tensordot, reduce_sum, concat
import numpy


class AdvAttentionLSTM(Model):
    def __init__(self, feature_dim, latent_dim, adv_eps, l2_norm_alpha, adv_beta):
        super().__init__()
        tf.random.set_seed(123456)
        self.attn_lstm_layer = AttnLSTM(feature_dim, latent_dim)
        self.adv_eps = adv_eps
        self.l2_norm_alpha = l2_norm_alpha
        self.adv_beta = adv_beta
        self.regular_predictions = None
        self.adversarial_predictions = None
        self.loss_fn = tf.losses.Hinge()

    def call(self, args, training):

        if training is True:

            input = args[0]
            output = args[1]

            # with tf.GradientTape() as tape:

            final_layer, predicted_output = self.attn_lstm_layer(
                input, training=True)

            normal_loss = self.get_normal_loss(predicted_output, output)

            self.regular_predictions = predicted_output

            return predicted_output

            # final_layer_gradient = tape.gradient(
            #     normal_loss, final_layer)[0]

            # tf.stop_gradient(final_layer_gradcient)

            # delta_adv = tf.nn.l2_normalize(final_layer_gradient)

            # adversarial_input = final_layer + delta_adv * self.adv_eps

            # self.adversarial_predictions = self.attn_lstm_layer.decoding_layer(
            #     adversarial_input)

    def get_normal_loss(self, predicted_output, correct_output):
        return self.loss_fn(predicted_output, correct_output)

    def get_total_loss(self, pred, correct_output):

        return self.loss_fn(pred, correct_output)

        loss = self.loss_fn(self.regular_predictions, correct_output)

        # l2_norm_loss = self.get_regluarization_loss() * self.l2_norm_alpha

        # adv_loss = self.loss_fn(
        #     self.adversarial_predictions, correct_output) * self.adv_beta

        total_loss = loss

        # total_loss = loss + l2_norm_loss + adv_loss

        return total_loss

    def get_regluarization_loss(self):
        return self.attn_lstm_layer.get_regluarization_loss()


class AttnLSTM(Model):

    def __init__(self, feature_dim, latent_dim):

        super().__init__()
        tf.random.set_seed(123456)

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

    def call(self, input, training):

        if training is True:

            encoded_feature = self.encoding_layer(input)

            rnn_output = self.rnn_layer(encoded_feature)

            attention_latent = tanh(
                tensordot(rnn_output, self.attention_weight, axes=1) + self.attention_bias)

            attention_score = tensordot(
                attention_latent, self.attention_u, axes=1)

            attention_score = tf.nn.softmax(attention_score)

            attention_output = reduce_sum(
                rnn_output * tf.expand_dims(attention_score, -1), 1)

            output = concat(
                [rnn_output[:, -1, :], attention_output], axis=1)

            predictions = self.decoding_layer(output)

            return output, predictions

    def get_regluarization_loss(self):
        loss = 0
        for train_var in self.trainable_variables:
            loss += tf.nn.l2_loss(train_var) * 0.001
        return tf.cast(loss, dtype=tf.float64)
