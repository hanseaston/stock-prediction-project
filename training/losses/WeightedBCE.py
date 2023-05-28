import tensorflow as tf


class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight):
        super(WeightedBinaryCrossentropy, self).__init__()
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        # Apply BinaryCrossentropy loss
        bce_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)(y_true, y_pred)

        # Apply class weights
        weighted_loss = tf.multiply(self.pos_weight, bce_loss)

        return weighted_loss
