from load_data import load_cla_data
from time import time
import argparse
import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops


class AdversarialLSTM:
    def __init__(self, data_path,
                 model_path,
                 model_save_path,
                 parameters,
                 steps=1,
                 epochs=50,
                 batch_size=256,
                 gpu=False,
                 tra_date='2014-01-02',
                 val_date='2015-08-03',
                 tes_date='2015-10-01',
                 att=0,
                 hinge=0,
                 fix_init=0,
                 adv=0,
                 reload=0):

        self.data_path = data_path
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.paras = copy.copy(parameters)

        # training parameters
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu

        if att == 1:
            self.att = True
        else:
            self.att = False

        if hinge == 1:
            self.hinge = True
        else:
            self.hinge = False

        if fix_init == 1:
            self.fix_init = True
        else:
            self.fix_init = False

        if adv == 1:
            self.adv_train = True
        else:
            self.adv_train = False

        if reload == 1:
            self.reload = True
        else:
            self.reload = False

        # load data
        self.tra_date = tra_date
        self.val_date = val_date
        self.tes_date = tes_date
        self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
                self.data_path,
                tra_date, val_date, tes_date, seq=self.paras['seq']
            )
        self.fea_dim = self.tra_pv.shape[2]
        print(self.fea_dim)

    def construct_graph(self):

        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'

        with tf.device(device_name):

            tf.reset_default_graph()

            # TODO: fix randomization
            if self.fix_init:
                tf.set_random_seed(123456)

            # ground truth outcomes (whether stock is going up or down), (N x 1)
            self.gt_var = tf.placeholder(tf.float32, [None, 1])

            # inputs, (N, S, D), where S is the sequence length, D is the feature dimension
            self.pv_var = tf.placeholder(
                tf.float32, [None, self.paras['seq'], self.fea_dim]
            )

            # weekday variable, (N, S, 5)
            self.wd_var = tf.placeholder(
                tf.float32, [None, self.paras['seq'], 5]
            )

            # the hidden dimension for the LSTM layer
            self.lstm_cell = tf.keras.layers.LSTMCell(units=self.paras['unit'])

            # self.outputs, _ = tf.nn.dynamic_rnn(
            #     # self.outputs, _ = tf.nn.static_rnn(
            #     self.lstm_cell, self.pv_var, dtype=tf.float32
            #     # , initial_state=ini_sta
            # )

            # feeding the input layer into latent dimension
            self.in_lat = tf.layers.dense(
                self.pv_var, units=self.fea_dim,
                activation=tf.nn.tanh, name='in_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            # feeding the latent dimensional input sequence into the rnn
            self.outputs, _ = tf.nn.dynamic_rnn(
                # self.outputs, _ = tf.nn.static_rnn(
                self.lstm_cell, self.in_lat, dtype=tf.float32
                # , initial_state=ini_sta
            )

            print(self.outputs.shape)

            self.loss = 0
            self.adv_loss = 0
            self.l2_norm = 0

            if self.att:
                with tf.variable_scope('lstm_att') as scope:

                    self.av_W = tf.get_variable(
                        name='att_W', dtype=tf.float32,
                        shape=[self.paras['unit'], self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )

                    self.av_b = tf.get_variable(
                        name='att_h', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.zeros_initializer()
                    )

                    self.av_u = tf.get_variable(
                        name='att_u', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )

                    self.a_laten = tf.tanh(
                        tf.tensordot(self.outputs, self.av_W,
                                     axes=1) + self.av_b)

                    self.a_scores = tf.tensordot(self.a_laten, self.av_u,
                                                 axes=1,
                                                 name='scores')

                    self.a_alphas = tf.nn.softmax(self.a_scores, name='alphas')

                    self.a_con = tf.reduce_sum(
                        self.outputs * tf.expand_dims(self.a_alphas, -1), 1)

                    self.fea_con = tf.concat(
                        [self.outputs[:, -1, :], self.a_con],
                        axis=1)

                    print('adversarial scope')

                    # training loss
                    self.pred = self.adv_part(self.fea_con)

                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(
                            self.gt_var, self.pred)

                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:

                        print('gradient noise')

                        self.delta_adv = tf.gradients(
                            self.loss, [self.fea_con])[0]

                        tf.stop_gradient(self.delta_adv)

                        self.delta_adv = tf.nn.l2_normalize(
                            self.delta_adv, axis=1)

                        self.adv_pv_var = self.fea_con + \
                            self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()

                        self.adv_pred = self.adv_part(self.adv_pv_var)

                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(
                                self.gt_var, self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(
                                self.gt_var, self.adv_pred)

            else:
                with tf.variable_scope('lstm_att') as scope:
                    print('adversarial scope')
                    # training loss
                    self.pred = self.adv_part(self.outputs[:, -1, :])
                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(
                            self.gt_var, self.pred)
                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:
                        print('gradient noise')
                        self.delta_adv = tf.gradients(
                            self.loss, [self.outputs[:, -1, :]])[0]
                        tf.stop_gradient(self.delta_adv)
                        self.delta_adv = tf.nn.l2_normalize(self.delta_adv,
                                                            axis=1)
                        self.adv_pv_var = self.outputs[:, -1, :] + \
                            self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()
                        self.adv_pred = self.adv_part(self.adv_pv_var)
                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(self.gt_var,
                                                                 self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(self.gt_var,
                                                               self.adv_pred)

            # regularizer
            self.tra_vars = tf.trainable_variables('lstm_att/pre_fc')
            for var in self.tra_vars:
                self.l2_norm += tf.nn.l2_loss(var)

            self.obj_func = self.loss + \
                self.paras['alp'] * self.l2_norm + \
                self.paras['bet'] * self.adv_loss

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.paras['lr']
            ).minimize(self.obj_func)

    def adv_part(self, adv_inputs):

        print('adversial part')

        if self.att:
            with tf.variable_scope('pre_fc'):

                self.fc_W = tf.get_variable(
                    'weights', dtype=tf.float32,
                    shape=[self.paras['unit'] * 2, 1],
                    initializer=tf.glorot_uniform_initializer()
                )

                self.fc_b = tf.get_variable(
                    'biases', dtype=tf.float32,
                    shape=[1, ],
                    initializer=tf.zeros_initializer()
                )

                if self.hinge:
                    pred = tf.nn.bias_add(
                        tf.matmul(adv_inputs, self.fc_W), self.fc_b
                    )
                else:
                    pred = tf.nn.sigmoid(
                        tf.nn.bias_add(tf.matmul(self.fea_con, self.fc_W),
                                       self.fc_b)
                    )
        else:
            # One hidden layer
            if self.hinge:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=None,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
            else:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=tf.nn.sigmoid,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
        return pred


if __name__ == '__main__':
    desc = 'the lstm model'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./dataset/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                        help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
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

    parameters = {
        'seq': int(args.seq),
        'unit': int(args.unit),
        'alp': float(args.alpha_l2),
        'bet': float(args.beta_adv),
        'eps': float(args.epsilon_adv),
        'lr': float(args.learning_rate)
    }

    # if 'stocknet' in args.path:
    #     tra_date = '2014-01-02'
    #     val_date = '2015-08-03'
    #     tes_date = '2015-10-01'

    # elif 'kdd17' in args.path:
    #     tra_date = '2007-01-03'
    #     val_date = '2015-01-02'
    #     tes_date = '2016-01-04'

    # else:
    #     print('unexpected path: %s' % args.path)
    #     exit(0)

    tra_date = '2014-01-02'
    val_date = '2015-08-03'
    tes_date = '2015-10-01'

    pure_LSTM = AdversarialLSTM(
        data_path=args.path,
        model_path=args.model_path,
        model_save_path=args.model_save_path,
        parameters=parameters,
        steps=args.step,
        epochs=args.epoch,
        batch_size=args.batch_size,
        gpu=args.gpu,
        tra_date=tra_date,
        val_date=val_date,
        tes_date=tes_date,
        att=args.att,
        hinge=args.hinge_lose,
        fix_init=args.fix_init,
        adv=args.adv,
        reload=args.reload
    )

    print("hey")

    pure_LSTM.construct_graph()

    # if args.action == 'train':
    #     pure_LSTM.train()
    # elif args.action == 'test':
    #     pure_LSTM.test()
    # elif args.action == 'report':
    #     for i in range(5):
    #         pure_LSTM.train()
    # elif args.action == 'pred':
    #     pure_LSTM.predict_record()
    # elif args.action == 'adv':
    #     pure_LSTM.predict_adv()
    # elif args.action == 'latent':
    #     pure_LSTM.get_latent_rep()
