from argparse import ArgumentParser


def parse_arguments():

    desc = 'stock analysis engine'

    parser = ArgumentParser(description=desc)

    parser.add_argument('-p', '--path', help='path of stock data', type=str,
                        default='./dataset/kdd17')

    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=10)

    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)

    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-5,
                        help='alpha for l2 regularizer')

    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')

    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                        help='epsilon to control the scale of noise')

    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)

    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=50)

    parser.add_argument('-r', '--learning_rate', help='learning rate',
                        type=float, default=0.001)

    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')

    args = parser.parse_args()

    return args
