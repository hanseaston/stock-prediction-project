import tensorflow as tf
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from ..dataset_construction.ternary_constructor import ternary_constructor
from ..models.ModelSelector import select_model

######### Training configuration #########
BATCH_SIZE = 1024
#######################################


def grid_search(params):

    batched_train_dataset = train_dataset.batch(batch_size)

    alphas = [0.001, 0.01]
    betas = [0.001, 0.05]
    eps = [0.01, 0.05]
    latent_dims = [16, 32]
    lrs = [0.01]

    best_model = None
    best_acc = 0.0

    for alpha, beta, ep, latent_dim, lr in itertools.product(alphas, betas, eps, latent_dims, lrs):

        print(f"trying alpha {alpha}, beta {beta}, ep {ep}, \
               latent dimension {latent_dim}, lr {lr}")

        model = AdvAttnLSTM(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            adv_eps=ep,
            l2_norm_alpha=alpha,
            adv_beta=beta
        )

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

        val_predictions = model(val_pv, training=False)
        val_predictions = (np.sign(val_predictions) + 1) / 2
        accuracy = accuracy_score(val_predictions, val_gt)
        print(f"val accuracy before training: {np.round(accuracy * 100, 2)}%")

        for _ in tqdm(range(num_epoch)):

            for step, (x_batch_train, y_batch_train) in enumerate(batched_train_dataset):

                with tf.GradientTape() as tape:
                    model([x_batch_train, y_batch_train], training=True)
                    loss_value = model.get_total_loss(y_batch_train)

                gradients = tape.gradient(
                    loss_value, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_weights))

        val_predictions = model(val_pv, training=False)
        val_predictions = (np.sign(val_predictions) + 1) / 2
        accuracy = accuracy_score(val_predictions, val_gt)

        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model

    test_predictions = best_model(tes_pv, training=False)
    test_predictions = (np.sign(test_predictions) + 1) / 2
    accuracy = accuracy_score(test_predictions, tes_gt)
    print(f"test accuracy with best model: {np.round(accuracy * 100, 2)}%")


if __name__ == '__main__':
    grid_search()
