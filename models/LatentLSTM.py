import tensorflow as tf
from keras.layers import Dense, RNN, LSTMCell
from keras.models import Model

######### Model configuration #########
LATENT_DIM = 32
L2_ALPHA = 0
#######################################


class LatentLSTM(Model):

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

        self.decoding_layer = Dense(units=output_dim,
                                    activation='sigmoid',
                                    kernel_initializer='glorot_uniform')

        self.loss_fn = loss_fn

    def call(self, input, training):

        encoded_feature = self.encoding_layer(input)

        rnn_output = self.rnn_layer(encoded_feature)

        predictions = self.decoding_layer(rnn_output[:, -1, :])

        return predictions

    def get_total_loss(self, correct_output, predicted_output):
        reg_loss = 0
        for train_var in self.trainable_variables:
            reg_loss += tf.nn.l2_loss(train_var)

        return self.loss_fn(correct_output, predicted_output) + reg_loss * self.l2_norm_alpha

     # def get_config(self):

    #     config = super().get_config()

    #     config['l2_norm_alpha'] = 1e-3

    #     config['encoding_layer'] = self.encoding_layer = Dense(units=13,
    #                                                            activation='tanh',
    #                                                            kernel_initializer='glorot_uniform')

    #     config['rnn_layer'] = RNN(LSTMCell(
    #         units=64,
    #         activation='tanh',
    #         kernel_initializer='glorot_uniform'),
    #         return_sequences=True)

    #     config['decoding_layer'] = Dense(units=1,
    #                                      activation='sigmoid',
    #                                      kernel_initializer='glorot_uniform')

    #     config['loss_fn'] = BinaryCrossentropy(from_logits=False)

    #     return config
