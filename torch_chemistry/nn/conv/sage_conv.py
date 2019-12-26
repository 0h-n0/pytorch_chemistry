import torch
from torch.nn import Parameter
import torch.nn.functional as F

from . import GNNConv
from ..init import feedforward_init


class SAGEconv(GNNConv):
    def __init__(self):
        pass

    def forward(self, nodes, edges):
        pass
