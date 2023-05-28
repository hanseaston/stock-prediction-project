import tensorflow as tf
from keras.layers import Dense, RNN, LSTMCell
from keras.models import Model
from keras.initializers import GlorotUniform, Zeros
from tensorflow import tanh, tensordot, reduce_sum, concat


######### Model configuration #########
LATENT_DIM = 32
L2_ALPHA = 1e-3
#######################################


class AttnLSTM(Model):

    def __init__(self, feature_dim, output_dim, loss_fn, **kwargs):

        super().__init__()

        self.l2_norm_alpha = L2_ALPHA

        self.encoding_layer = Dense(units=feature_dim,
                                    activation='tanh',
                                    kernel_initializer='glorot_uniform')

        if 'latent_dim' in kwargs:
            latent_dim = kwargs['latent_dim']
        else:
            latent_dim = LATENT_DIM

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

        self.attention_u = tf.Variable(
            initial_value=GlorotUniform()((latent_dim, )),
            dtype=tf.float32,
            name='attention_u')

        self.decoding_layer = Dense(units=output_dim,
                                    activation='sigmoid',
                                    kernel_initializer='glorot_uniform')

        self.loss_fn = loss_fn

    def call(self, args, training):

        input = args[0] if training else args

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

        return predictions

    def get_total_loss(self, correct_output, predicted_output):
        reg_loss = 0
        for train_var in self.trainable_variables:
            reg_loss += tf.nn.l2_loss(train_var)

        return self.loss_fn(correct_output, predicted_output) + reg_loss * self.l2_norm_alpha
