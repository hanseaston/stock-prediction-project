import tensorflow as tf


class WeightedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, cost_matrix):
        super().__init__()
        self.cost_matrix = cost_matrix

    def call(self, y_true, y_pred):

        base_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)

        # Get the true and predicted class indices
        true_class_indices = tf.argmax(y_true, axis=-1)
        predicted_class_indices = tf.argmax(y_pred, axis=-1)

        # Gather the corresponding costs
        rows = tf.range(tf.shape(y_true)[0])
        indices = tf.stack([rows, predicted_class_indices,
                           true_class_indices], axis=1)
        costs = tf.gather_nd(self.cost_matrix, indices)

        # Apply the costs to the base loss
        weighted_loss = base_loss * costs

        return tf.reduce_mean(weighted_loss)
