import torch
from torch.nn import Parameter

from . import GNNConv
from .utils import batched_sparse_eyes
from ..init import uniform


class GCNConv(GNNConv):
    def __init__(self, in_channels, out_channels,
                 bias=True, normalize=True, **kwargs):
        '''
        This module is modified for a datset inculding batchsize dimension. Not support sparse matrix manupulation.
        ref) "https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv"

        '''
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x: torch.FloatTensor, edges: torch.FloatTensor):
        x = x.matmul(self.weight)
        batch_size, node_size, _ = edges.size()
        identity = batched_sparse_eyes(node_size, batch_size, x.dtype, edges.device).to_dense()
        densed_edges = edges.float()
        adj_tilda = densed_edges.add(identity)
        if self.normalize:
            inv_sqrt_degree_mat = 1 / torch.sqrt(torch.diag_embed(densed_edges.sum(axis=2)))
            inv_sqrt_degree_mat[inv_sqrt_degree_mat == float('inf')] = 0
            adj_tilda = adj_tilda.matmul(inv_sqrt_degree_mat)
            adj_tilda = inv_sqrt_degree_mat.matmul(adj_tilda)
        x = adj_tilda.matmul(x)
        return x
