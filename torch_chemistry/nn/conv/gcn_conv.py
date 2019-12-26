import torch
from torch.nn import Parameter

from . import GNNConv
from ..init import uniform


class GCNConv(GNNConv):
    def __init__(self, in_channels, out_channels,
                 bias=True, **kwargs):
        '''
        This module is modified for a datset inculding batchsize dimension. Not support sparse matrix manupulation.
        ref) "https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv"

        '''
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        if self.bias is not None:
             self.bias.data = torch.zeros_like(self.bias)

    def forward(self, nodes: torch.FloatTensor, edges: torch.FloatTensor):
        '''
        edges: support densed edge

        '''
        h = nodes.matmul(self.weight)
        if self.bias is not None:
             h = h + self.bias
        adj = self._laplacianize(edges)
        print(adj.shape)
        h = adj.matmul(h)
        return h
