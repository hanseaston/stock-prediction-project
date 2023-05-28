from models.LSTM import LSTM
from models.AdvAttnLSTM import AdvAttnLSTM
from models.NeuralNets import NeuralNets
from models.AttnLSTM import AttnLSTM


def select_model(model_name, args):
    if model_name == 'LSTM':
        return LSTM(**args)
    if model_name == 'AttnLSTM':
        return AttnLSTM(**args)
    if model_name == 'AdvAttnLSTM':
        return AdvAttnLSTM(**args)
    if model_name == 'NeuralNets':
        return NeuralNets(**args)
