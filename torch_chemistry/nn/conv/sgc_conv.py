import torch
from torch.nn import Parameter

from . import GNNConv
from ..init import uniform


class SGCConv(GNNConv):
    def __init__(self, in_channels, out_channels,
                 bias=True, **kwargs):
        pass
