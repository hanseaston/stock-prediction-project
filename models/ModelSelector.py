from models.LSTM import LSTM
from models.AdvAttnLSTM import AdvAttnLSTM
from models.NeuralNets import NeuralNets


def select_model(model_name, args):
    if model_name == 'LSTM':
        return LSTM(**args)
    if model_name == 'AdvAttnLSTM':
        return AdvAttnLSTM(**args)
    if model_name == 'NeuralNets':
        return NeuralNets(**args)
