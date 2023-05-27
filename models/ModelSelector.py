from LSTM import LSTM
from AdvAttnLSTM import AdvAttnLSTM
from NeuralNets import NeuralNets


def select_model(model_name, args):
    if model_name == 'LSTM':
        return LSTM(**args)
    if model_name == 'AdvAttnLSTM':
        return AdvAttnLSTM(**args)
    if model_name == 'NeuralNets':
        return NeuralNets(**args)
