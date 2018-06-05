import torch
import torch.nn as nn
from torch.autograd import Variable

from e2edd import MAX_ATOMIC_NUM
from pytorch_chemistry.models.basic import RNNModel
from pytorch_chemistry.models.graph_basic import EmbedAtomID
from pytorch_chemistry.models.graph_basic import GraphLinear

class GGNN(nn.Module):
    """Gated Graph Neural Networks (GGNN)

    See: Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015).\
        Gated graph sequence neural networks. \
        `arXiv:1511.05493 <https://arxiv.org/abs/1511.05493>`_

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying or not

    """
    NUM_EDGE_TYPE = 4    
    def __init__(self,
                 out_dim,
                 hidden_dim=16,
                 n_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM,
                 concat_hidden=False,
                 weight_tying=True,
                 rnn_type='GRU',
                 config_glayer={},
                 config_rnn={}
    ):
        super(GGNN, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        self.embed = EmbedAtomID(embedding_dim=hidden_dim,
                                 num_embeddings=n_atom_types)
        self.message_layers = nn.ModuleList(
            [GraphLinear(hidden_dim,
                         self.NUM_EDGE_TYPE * hidden_dim,
                         **config_glayer)
             for _ in range(n_message_layer)]
            )
        self.update_layer = RNNModel(2 * hidden_dim,
                                     hidden_dim,
                                     rnn_type,
                                     **config_rnn
                                 )
        self.i_layers = nn.ModuleList(
            [GraphLinear(2 * hidden_dim,
                         out_dim,
                         **config_glayer)
             for _ in range(n_readout_layer)]
            )
        self.j_layers = nn.ModuleList(
            [GraphLinear(hidden_dim,
                         out_dim,
                         **config_glayer)
             for _ in range(n_readout_layer)]
            )

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def update(self, x, adj, step=0):
        # --- Message & Update part ---
        # (minibatch, atom, ch)
        h = x
        mb, atom, ch = h.shape
        out_ch = ch
        message_layer_index = 0 if self.weight_tying else step
        m = self.message_layers[message_layer_index](h)
        
        m = m.view(mb, atom, out_ch, self.NUM_EDGE_TYPE)
        
        # m: (minibatch, ch, atom, edge_type)
        # Transpose
        m = torch.transpose(m, 1, 3)
        # m: (minibatch, edge_type, atom, ch)

        adj = adj.view(mb * self.NUM_EDGE_TYPE, atom, atom)
        # (minibatch * edge_type, atom, out_ch)
        m = m.contiguous()
        m = m.view(mb * self.NUM_EDGE_TYPE, atom, out_ch)
        
        m = torch.matmul(adj, m)

        # (minibatch * edge_type, atom, out_ch)
        m = m.view(mb, self.NUM_EDGE_TYPE, atom, out_ch)
        # Take sum
        m = torch.sum(m, dim=1)
        # (minibatch, atom, out_ch)

        # --- Update part ---
        # Contraction
        h = h.view(mb * atom, ch)

        # Contraction
        m = m.view(mb * atom, ch)
        o = torch.cat((h, m), dim=1).view(1, mb * atom, 2 * ch)

        out_h, _ = self.update_layer(o)
        # Expansion
        out_h = out_h.view(mb, atom, ch)
        return out_h

    def readout(self, h, h0, step=0):
         # --- Readout part ---
        index = step if self.concat_hidden else 0
        # h, h0: (minibatch, atom, ch)

        g = nn.functional.sigmoid(
            self.i_layers[index](torch.cat((h, h0), dim=2))) \
            * self.j_layers[index](h)

        g = torch.sum(g, dim=1)  # sum along atom's axis
        return g

    def forward(self, atom_array, adj):
        """Forward propagation
        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix with edge-type
                information
        Returns:
            ~torch.autograd.Variable: minibatch of fingerprint
        """
        assert isinstance(atom_array, Variable), ('must be Variable. ')
        assert isinstance(adj, Variable), ('must be Variable. ')
        
        h = self.embed(atom_array)  # (minibatch, max_num_atoms)

        h0 = h.clone()
        g_list = []

        for step in range(self.n_layers):
            h = self.update(h, adj, step)
            if self.concat_hidden:
                g = self.readout(h, h0, step)
                g_list.append(g)

        if self.concat_hidden:
            return torch.cat(g_list, dim=1)
        else:
            g = self.readout(h, h0, 0)
            return g

        
if __name__ == '__main__':
    g = GGNN(3)
    print(g)
