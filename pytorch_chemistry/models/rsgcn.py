"""
Implementation of Renormalized Spectral Graph Convolutional Network (RSGCN)
See: Thomas N. Kipf and Max Welling, \
    Semi-Supervised Classification with Graph Convolutional Networks. \
    September 2016. \
    `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_
"""
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from e2edd import MAX_ATOMIC_NUM
from pytorch_chemistry.models.basic import RNNModel
from pytorch_chemistry.models.graph_basic import EmbedAtomID
from pytorch_chemistry.models.graph_basic import GraphLinear


class RSGCNUpdate(nn.Module):
    """RSGCN sub module for message and update part
    Args:
        in_features (int): input channel dimension
        out_features (int): output channel dimension
    """    
    def __init__(self, in_features, out_features):
        super(RSGCNUpdate, self).__init__()
        self.graph_linear = GraphLinear(
            in_features,
            out_features,
            bias=False)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, w_adj):
        x = torch.matmul(w_adj, x)
        return self.graph_linear(x)


class RSGCNBlock(nn.Module):
    """RSGCN sub module for Graph convolution.
        in_features (int): input channel dimension
        out_features (int): output channel dimension
        use_batch_norm (bool): Batch Normalization
        dropout (float): dropout value
    """    
    def __init__(self, in_features, out_features,
                 use_batch_norm=True, dropout=0.2):
        super(RSGCNBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_batch_norm = use_batch_norm
        self.gconv = RSGCNUpdate(in_features, out_features)
        self.dropout = dropout
        if use_batch_norm:
            self.gbn = GraphBatchNorm(out_features)
        if dropout != 0.0:
            self.dp = nn.Dropout()
        
    def forward(self, x, w_adj):
        x = self.gconv(x, w_adj)
        if self.use_batch_norm:
            x = self.gbn(x)
        if self.dropout != 0.0:
            x = self.dp(x)
        return x

    
def rsgcn_readout_sum(x: Variable, activation: str=None) -> Variable:
    """Default readout function for `RSGCN`
    Args:
        x (torch.autograd.Variable): shape consists of (minibatch, atom, ch).
        activation: activation function, default is `None`.
            You may consider taking other activations, for example `sigmoid`,
            `relu` or `softmax` along `axis=2` (ch axis) etc.
    Returns: result of readout, its shape should be (minibatch, out_ch)
    """
    if activation is not None:
        h = activation(x)
    else:
        h = x
    y = torch.sum(h, dim=1)  # sum along node axis
    return y


class RSGCN(nn.Module):
    """Renormalized Spectral Graph Convolutional Network (RSGCN)
    See: Thomas N. Kipf and Max Welling, \
        Semi-Supervised Classification with Graph Convolutional Networks. \
        September 2016. \
        `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_
    The name of this model "Renormalized Spectral Graph Convolutional Network
    (RSGCN)" is named by us rather than the authors of the paper above.
    The authors call this model just "Graph Convolution Network (GCN)", but
    we think that "GCN" is bit too general and may cause namespace issue.
    That is why we did not name this model as GCN.
    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_atom_types (int): number of types of atoms
        n_layers (int): number of layers
        use_batch_norm (bool): If True, batch normalization is applied after
            graph convolution.
        readout (Callable): readout function. If None, `rsgcn_readout_sum` is
            used. To the best of our knowledge, the paper of RSGCN model does
            not give any suggestion on readout.
    """    
    def __init__(self, out_dim, hidden_dim=64, n_layers=3,
                 n_atom_types=MAX_ATOMIC_NUM,
                 use_batch_norm=False,
                 readout: Callable[[Variable,], Variable]=None):
        super(RSGCN, self).__init__()
        
        in_dims = [hidden_dim for _ in range(n_layers)]
        out_dims = [hidden_dim for _ in range(n_layers)]
        out_dims[n_layers - 1] = out_dim
        
        self.embed = EmbedAtomID(embedding_dim=hidden_dim)
        
        self.gconvs = nn.ModuleList(
            [RSGCNBlock(in_dims[i], out_dims[i], use_batch_norm, )
             for i in range(n_layers)])
        
        if isinstance(readout, nn.Module):
                self.readout = readout
        if not isinstance(readout, nn.Module):
            self.readout = readout or rsgcn_readout_sum
            
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
    def __call__(self, graph, adj):
        """Forward propagation
        Args:
            graph (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix
                `adj[mol_index]` represents `mol_index`-th molecule's
                adjacency matrix
        Returns:
            ~torch.autograd.Variable: minibatch of fingerprint
        """
        h = self.embed(graph)

        # h: (minibatch, nodes, ch)

        if isinstance(adj, Variable):
            w_adj = adj.data
        else:
            w_adj = adj
        w_adj = Variable(w_adj, requires_grad=False)

        # --- RSGCN update ---
        for i, gconv in enumerate(self.gconvs):
            h = gconv(h, w_adj)
            if i < self.n_layers - 1:
                h = F.relu(h)

        # --- readout ---
        y = self.readout(h)
        return y    

if __name__ == '__main__':
    r = RSGCN(30)
    print(r)
