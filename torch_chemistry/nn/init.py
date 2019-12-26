import math

import torch.nn as nn


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def feedforward_init(dnn: nn.Module,
                     init_mean=0,
                     init_std=1,
                     init_xavier: bool=True,
                     init_normal: bool=True,
                     init_gain: float=1.0):
    for name, p in dnn.named_parameters():
        if 'bias' in name:
            p.data.zero_()
        if 'weight' in name:
            if len(p.data.shape) == 1:
                if init_normal:
                    nn.init.normal_(p.data, init_gain)
                else:
                    nn.init.uniform_(p.data, init_gain)
                continue
            if init_xavier:
                print(p.data.shape)
                if init_normal:
                    nn.init.xavier_normal(p.data, init_gain)
                else:
                    nn.init.xavier_uniform(p.data, init_gain)
            else:
                if init_normal:
                    nn.init.normal(p.data, init_gain)
                else:
                    nn.init.uniform(p.data, init_gain)
