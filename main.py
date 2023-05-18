import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from Model import AdvAttentionLSTM
from load_data import load_cla_data
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train(params):

    tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt, \
        val_gt, tes_pv, tes_wd, tes_gt, feature_dim, latent_dim, alpha, beta, \
        eps, lr, batch_size, num_epoch = params

    train_dataset = tf.data.Dataset.from_tensor_slices((tra_pv, tra_gt))
    train_dataset = train_dataset.shuffle(buffer_size=len(tra_pv))
    batched_train_dataset = train_dataset.batch(batch_size)

    model = AdvAttentionLSTM(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        adv_eps=eps,
        l2_norm_alpha=alpha,
        adv_beta=beta
    )

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    # train_predictions = model(tra_pv, training=False)
    # train_predictions = (np.sign(train_predictions) + 1) / 2
    # accuracy = accuracy_score(train_predictions, tra_gt)
    # print(f"accuracy before training: {np.round(accuracy, 2)}%")

    test_predictions = model(tes_pv, training=False)
    test_predictions = (np.sign(test_predictions) + 1) / 2
    accuracy = accuracy_score(test_predictions, tes_gt)
    print(f"accuracy before training: {np.round(accuracy, 2)}%")

    for epoch in tqdm(range(num_epoch)):

        epoch_loss = 0

        for step, (x_batch_train, y_batch_train) in enumerate(batched_train_dataset):
            with tf.GradientTape() as tape:
                model([x_batch_train, y_batch_train], training=True)
                loss_value = model.get_total_loss(y_batch_train)
                epoch_loss += loss_value

            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # with tf.GradientTape() as tape:
        #     regularization_loss = model.get_regluarization_loss()
        #     gradients = tape.gradient(
        #         regularization_loss, model.trainable_variables)
        #     optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print(epoch_loss.numpy().item())

    # train_predictions = model(tra_pv, training=False)
    # train_predictions = (np.sign(train_predictions) + 1) / 2
    # accuracy = accuracy_score(train_predictions, tra_gt)
    # print(f"accuracy after training: {np.round(accuracy, 2)}%")

    test_predictions = model(tes_pv, training=False)
    test_predictions = (np.sign(test_predictions) + 1) / 2
    accuracy = accuracy_score(test_predictions, tes_gt)
    print(f"accuracy after training: {np.round(accuracy, 2)}%")


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

    tra_pv, tra_wd, tra_gt, \
        val_pv, val_wd, val_gt, \
        tes_pv, tes_wd, tes_gt = load_cla_data(data_path,
                                               tra_date, val_date, tes_date, seq=sequence_len)

    feature_dim = tra_pv.shape[2]

    # TODO: too many parameters, reorganize
    return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt, \
        val_gt, tes_pv, tes_wd, tes_gt, feature_dim, latent_dim, alpha, beta, eps, lr, batch_size, \
        num_epoch


def parse_arguments():

    desc = 'the lstm model'

    # TODO: organize arg parsers
    parser = ArgumentParser(description=desc)

    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./dataset/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=4)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                        help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=5e-2,
                        help='epsilon to control the scale of noise')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                        type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate',
                        type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-q', '--model_path', help='path to load model',
                        type=str, default='./saved_model/acl18_alstm/exp')
    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model',
                        default='./tmp/model')
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, pred')
    parser.add_argument('-m', '--model', type=str, default='pure_lstm',
                        help='pure_lstm, di_lstm, att_lstm, week_lstm, aw_lstm')
    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-a', '--att', type=int, default=1,
                        help='use attention model')
    parser.add_argument('-w', '--week', type=int, default=0,
                        help='use week day data')
    parser.add_argument('-v', '--adv', type=int, default=0,
                        help='adversarial training')
    parser.add_argument('-hi', '--hinge_lose', type=int, default=1,
                        help='use hinge lose')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    params = get_params(args)
    train(params)
