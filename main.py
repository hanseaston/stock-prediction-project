import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from load_data import load_cla_data
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools

# models
from LSTM import LSTM
from AdvAttnLSTM import AdvAttnLSTM
from NeuralNets import NeuralNets


def grid_search(params):
    tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt, \
        feature_dim, latent_dim, alpha, beta, \
        eps, lr, batch_size, num_epoch = params

    train_dataset = tf.data.Dataset.from_tensor_slices((tra_pv, tra_gt))
    train_dataset = train_dataset.shuffle(buffer_size=len(tra_pv))
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


def single_shot(params):
    tra_pv, tra_gt, _, _, tes_pv, tes_gt, \
        feature_dim, latent_dim, alpha, beta, \
        eps, lr, batch_size, num_epoch, sequence_len = params

    train_dataset = tf.data.Dataset.from_tensor_slices((tra_pv, tra_gt))
    train_dataset = train_dataset.shuffle(buffer_size=len(tra_pv))
    batched_train_dataset = train_dataset.batch(batch_size)

    args = {
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "l2_norm_alpha": alpha,
        "adv_beta": beta,
        "adv_eps": eps,
        "sequence_len": sequence_len
    }

    # model = get_model('LSTM', args)
    model = get_model('AdvAttnLSTM', args)
    # model = get_model('NeuralNets', args)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    for _ in tqdm(range(num_epoch)):

        epoch_loss = 0

        for _, (x_batch_train, y_batch_train) in enumerate(batched_train_dataset):

            with tf.GradientTape() as tape:

                predicted_output = model(
                    [x_batch_train, y_batch_train], training=True)
                loss_value = model.get_total_loss(
                    y_batch_train, predicted_output)
                epoch_loss += loss_value

            gradients = tape.gradient(
                loss_value, model.trainable_variables)

            optimizer.apply_gradients(
                zip(gradients, model.trainable_weights))

        # for each epoch, report train accuracy
        train_predictions = model(tra_pv, training=False)
        train_predictions = (np.sign(train_predictions) + 1) / 2
        accuracy = accuracy_score(train_predictions, tra_gt)
        print(
            f"train accuracy with current model: {np.round(accuracy * 100, 2)}%")

        # for each epoch, report test accuracy
        test_predictions = model(tes_pv, training=False)
        test_predictions = (np.sign(test_predictions) + 1) / 2
        accuracy = accuracy_score(test_predictions, tes_gt)
        print(
            f"test accuracy with current model: {np.round(accuracy * 100, 2)}%")

        print(f"loss: {epoch_loss}")

    # final test accuracy after training all epochs
    test_predictions = model(tes_pv, training=False)
    test_predictions = (np.sign(test_predictions) + 1) / 2
    accuracy = accuracy_score(test_predictions, tes_gt)
    print(f"test accuracy with best model: {np.round(accuracy * 100, 2)}%")


def train(params):

    tune_params = False

    if tune_params:
        grid_search(params)
    else:
        single_shot(params)


def get_model(model_name, args):
    if model_name == 'LSTM':
        return LSTM(**args)
    if model_name == 'AdvAttnLSTM':
        return AdvAttnLSTM(**args)
    if model_name == 'NeuralNets':
        return NeuralNets(**args)


def get_params(args):

    # hardcoding the date for now since we only use one dataset
    tra_date = '2014-01-02'
    val_date = '2015-08-03'
    tes_date = '2015-10-01'

    data_path = args.path
    batch_size = args.batch_size
    num_epoch = args.epoch
    sequence_len = int(args.seq)
    latent_dim = int(args.unit)
    alpha = float(args.alpha_l2)
    beta = float(args.beta_adv)
    eps = float(args.epsilon_adv)
    lr = float(args.learning_rate)

    tra_pv, _, tra_gt, \
        val_pv, _, val_gt, \
        tes_pv, _, tes_gt = load_cla_data(data_path,
                                          tra_date, val_date, tes_date, seq=sequence_len)

    feature_dim = tra_pv.shape[2]

    return tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt, \
        feature_dim, latent_dim, alpha, beta, eps, lr, batch_size, \
        num_epoch, sequence_len


def parse_arguments():

    desc = 'stock analysis engine'

    parser = ArgumentParser(description=desc)

    parser.add_argument('-p', '--path', help='path of stock data', type=str,
                        default='./dataset/kdd17')

    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=10)

    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)

    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-3,
                        help='alpha for l2 regularizer')

    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')

    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                        help='epsilon to control the scale of noise')

    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)

    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=100)

    parser.add_argument('-r', '--learning_rate', help='learning rate',
                        type=float, default=0.001)

    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    params = get_params(args)
    train(params)
