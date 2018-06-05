from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

def init_feedforward_weights(dnn: nn.Module,
                             init_mean=0,
                             init_std=1,
                             init_xavier: bool=True,
                             init_normal: bool=True,
                             init_gain: float=1.0):
    for name, p in dnn.named_parameters():
        if 'bias' in name:
            p.data.zero_()
        if 'weight' in name:            
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal(p.data, init_gain)
                else:
                    nn.init.xavier_uniform(p.data, init_gain)
            else:
                if init_normal:
                    nn.init.normal(p.data, init_gain)
                else:
                    nn.init.uniform(p.data, init_gain)

def _init_rnn_weights(rnn: nn.Module,
                      init_xavier: bool=True,
                      init_normal: bool=True,
                      init_gain: float=1.0,
                      ):
    for name, p in rnn.named_parameters():
        if 'bias' in name:
            p.data.fill_(0)
            if isinstance(rnn, (torch.nn.LSTM, torch.nn.LSTMCell)):
                n = p.nelement()
                p.data[n // 4:n // 2].fill_(1)  # forget bias
        elif 'weight' in name:
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal(p, init_gain)
                else:
                    nn.init.xavier_uniform(p, init_gain)
            else:
                if init_normal:
                    nn.init.normal(p, init_gain)
                else:
                    nn.init.uniform(p, init_gain)

class LinearModel(nn.Module):
    """Simple Linear Model
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 activation='relu',
                 init_mean=0,
                 init_std=1,
                 init_xavier: bool=True,
                 init_normal: bool=True,
                 init_gain: float=1.0):
        super(LinearModel, self).__init__()        
        
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias)
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
            
        init_feedforward_weights(self.fc,
                                 init_mean,
                                 init_std,
                                 init_xavier,
                                 init_normal,
                                 init_gain)
                                 
    def forward(self, x):
        if self.activation is not None:
            x = self.activation(self.fc(x))
        else:
            x = self.fc(x)
        return x
        
        
                    
class RNNModel(nn.Module):
    """Simple RNNmodel
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_type='LSTM',                 
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.0,
                 batch_first=True,
                 init_xavier: bool=True,
                 init_normal: bool=True,
                 init_gain: float=1.0,
                 concat: bool=True,
                 ):
        super(RNNModel, self).__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.concat = concat
        self.rnn_type = rnn_type
        if not self.rnn_type in ['LSTM', 'GRU', 'RNN']:
            raise NotImplementedError(self.rnn_type)
        self.rnn =\
                   getattr(nn, rnn_type)(input_size,
                                         hidden_size,
                                         num_layers,
                                         bidirectional=bidirectional,
                                         dropout=dropout,
                                         batch_first=batch_first)
        _init_rnn_weights(self.rnn,
                          init_xavier=init_xavier,
                          init_normal=init_normal,
                          init_gain=init_gain
                          )

    def forward(self,
                x: Union[Variable, PackedSequence],
                hx: Union[Variable, Tuple[Variable, ...]]=None) ->\
                Tuple[Variable, Variable]:
        
        assert isinstance(x, Variable) or\
            isinstance(x, PackedSequence), type(x)
        
        self.rnn.flatten_parameters()
        output, hx = self.rnn(x, hx)
        self.rnn.flatten_parameters()
        
        if (not self.concat) and self.bidirectional:
            B, T, F = output.size()
            output = output[:, :, :F//2] + output[:, :, F//2:]
            
        return output, hx                    


class LinearLayer(nn.Module):
    def __init__(self, data_shape: list, n_channel, n_layer):
        super(LinearLayer, self).__init__()        
        self.data_shape = data_shape
        assert len(self.data_shape) == 3, ('data_shape must be '
                                           '(batch_size, num_atom, features)')
        assert n_layer >= 0, 'n_layers must be oever 0.'        
        _, _, F = data_shape
        
        super(LinearLayer, self).__init__()

        if n_layer == 1:
            self.ll = LinearModel(F, n_channel)
        else:
            self.ll = nn.Sequential(
                LinearModel(F, n_channel, activation='relu'),
                *[LinearModel(n_channel, n_channel, activation='relu')
                  for _ in range(n_layer - 1) ]
                )
        
    def forward(self, x):
        n_batch, n_atom, n_channel = x.shape
        x = functions.reshape(x, (n_batch * n_atom, n_channel))
        for l in self.layers:
            x = l(x)
        x = functions.reshape(x, (n_batch, n_atom, self.n_output_channel))
        return x


if __name__ == '__main__':
    B = 10
    N = 100
    C = 33
    x = torch.autograd.Variable(torch.randn(B, N, C))
    ll = LinearLayer(x.size(), 4, 3)
    print(ll)
