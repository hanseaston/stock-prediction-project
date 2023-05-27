import tensorflow as tf


class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        loss = -y_true * tf.math.log(y_pred)
        loss = loss * self.weights
        return tf.reduce_sum(loss)
