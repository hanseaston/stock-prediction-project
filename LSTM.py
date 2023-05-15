import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias=0.1,
    ):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.size = size
        self.size_layer = size_layer
        self.output_size = output_size
        self.forget_bias = forget_bias

        # TODO: understand #StackedRNNCells
        self.rnn_cells = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(size_layer) for _ in range(num_layers)]
        )

        self.dense = tf.keras.layers.Dense(output_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, initial_state=None):
        if initial_state is None:
            initial_state = self.rnn_cells.get_initial_state(inputs)
        outputs, last_state = tf.nn.dynamic_rnn(
            self.rnn_cells, inputs, initial_state=initial_state, dtype=tf.float32
        )
        logits = self.dense(outputs[:, -1, :])
        return logits, last_state


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer
