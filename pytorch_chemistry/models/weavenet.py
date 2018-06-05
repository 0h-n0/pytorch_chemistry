import torch
import torch.nn as nn

from e2edd import MAX_ATOMIC_NUM
from pytorch_chemistry.models.basic import LinearLayer
from pytorch_chemistry.models.basic import init_feedforward_weights
from pytorch_chemistry.data.preprocessors.weavenet_preprocessor import \
    DEFAULT_NUM_MAX_ATOMS

def readout(a, model='sum', dim=1):
    if mode == 'sum':
        a = torch.sum(a, dim=dim)
    elif mode == 'max':
        a = torch.max(a, dim=dim)
    elif mode == 'summax':
        a_sum = torch.sum(a, dim=dim)
        a_max = torch.max(a, dim=dim)
        a = torch.cat((a_sum, a_max), dim=dim)
    else:
        raise ValueError('mode {} is not supported'.format(mode))
    return a


class AtomToPair(nn.Module):
    def __init__(self, data_shape, n_channel, n_layer, n_atom):
        super(AtomToPair, self).__init__()
        self.linear_layers = LinearLayer(data_shape, n_channel, n_layer)
        self.n_atom = n_atom
        self.n_channel = n_channel

    def forward(self, x:torch.autograd.Variable):
        n_batch, n_atom, n_feature = x.size()
        atom_repeat = x.view(n_batch, 1, n_atom, n_feature)
        atom_repeat = torch.stack([atom_repeat.clone() for _ in range(n_atom)], dim=1)
        atom_repeat = atom_repeat.view(n_batch, n_atom * n_atom, n_feature)
        
        atom_tile = functions.reshape(x, (n_batch, n_atom, 1, n_feature))
        atom_tile = torch.stack([atom_tile.clone() for _ in range(n_atom)], dim=2)
        atom_tile = atom_tile.view(n_batch, n_atom * n_atom, n_feature)        

        pair_x0 = torch.cat((atom_tile, atom_repeat), dim=2)
        pair_x0 = pair_x0.view(n_batch * n_atom * n_atom, n_feature * 2)
        
        for l in self.linear_layers:
            pair_x0 = l(pair_x0)
            
        pair_x0 = pair_x0.view(n_batch, n_atom * n_atom, self.n_channel)

        pair_x1 = torch.cat((atom_repeat, atom_tile), dim=2)
        pair_x1 = pair_x1.view(n_batch * n_atom * n_atom, n_feature * 2)
        
        for l in self.linear_layers:
            pair_x1 = l(pair_x1)
        pair_x1 = pair_x1.view(n_batch, n_atom * n_atom, self.n_channel)
        
        return pair_x0 + pair_x1


class PairToAtom(nn.Module):
    def __init__(self, data_shape, n_channel, n_layer, n_atom, mode='sum'):
        super(PairToAtom, self).__init__()
        self.linear_layers = LinearLayer(data_shape, n_channel, n_layer)        
        self.n_atom = n_atom
        self.n_channel = n_channel
        self.mode = mode

    def forward(self, x:torch.autograd.Variable):
        n_batch, n_pair, n_feature = x.size()
        a = x.view(n_batch * (self.n_atom * self.n_atom), n_feature)
        
        for l in self.linearLayer:
            a = l(a)
        a = a.view(n_batch, self.n_atom, self.n_atom, self.n_channel)
        a = readout(a, mode=self.mode, axis=2)
        return a
    

class WeaveModule(nn.Module):
    def __init__(self, data_shape, n_atom, output_channel, n_sub_layer,
                 readout_mode='sum'):
        super(WeaveModule, self).__init__()
        self.atom_layer = LinearLayer(data_shape, output_channel, n_sub_layer)
        self.pair_layer = LinearLayer(data_shape, output_channel, n_sub_layer)
        self.atom_to_atom = LinearLayer(data_shape, output_channel, n_sub_layer)
        self.pair_to_pair = LinearLayer(data_shape, output_channel, n_sub_layer)
        self.atom_to_pair = AtomToPair(data_shape, output_channel,
                                       n_sub_layer, n_atom)
        self.pair_to_atom = PairToAtom(data_shape, output_channel, n_sub_layer,
                                       n_atom, mode=readout_mode)
        self.n_atom = n_atom
        self.n_channel = output_channel
        self.readout_mode = readout_mode

    def forward(self, atom_x, pair_x, atom_only=False):
        a0 = self.atom_to_atom(atom_x)
        a1 = self.pair_to_atom(pair_x)
        a = torch.cat([a0, a1], axis=2)
        next_atom = self.atom_layer(a)
        if atom_only:
            return next_atom

        p0 = self.atom_to_pair(atom_x)
        p1 = self.pair_to_pair(pair_x)
        p = torch.cat([p0, p1], axis=2)
        next_pair = self.pair_layer(p)
        return next_atom, next_pair


class WeaveNet(nn.Module):
    """WeaveNet implementation
    Args:
        weave_channels (list): list of int, output dimension for each weave
            module
        hidden_dim (int): hidden dim
        n_atom (int): number of atom of input array
        n_sub_layer (int): number of layer for each `AtomToPair`, `PairToAtom`
            layer
        n_atom_types (int): number of atom id
        readout_mode (str): 'sum' or 'max' or 'summax'
    """

    def __init__(self, data_shape, weave_channels=None, hidden_dim=16,
                 n_atom=DEFAULT_NUM_MAX_ATOMS,
                 n_sub_layer=1, n_atom_types=MAX_ATOMIC_NUM,
                 readout_mode='sum'):
        weave_channels = weave_channels or WEAVENET_DEFAULT_WEAVE_CHANNELS
        weave_module = [
            WeaveModule(data_shape, n_atom, c, n_sub_layer,
                        readout_mode=readout_mode)
            for c in weave_channels
        ]

        super(WeaveNet, self).__init__()
        self.embed = EmbedAtomID(embedding_dim=hidden_dim,
                                 num_embeddings=n_atom_types)
        self.weave_module = nn.ModuleList(weave_module)

        self.readout_mode = readout_mode

    def forward(self, atom_x, pair_x, train=True):
        if atom_x.dtype == self.xp.int32:
            # atom_array: (minibatch, atom)
            atom_x = self.embed(atom_x)

        for i in range(len(self.weave_module)):
            if i == len(self.weave_module) - 1:
                # last layer, only `atom_x` is needed.
                atom_x = self.weave_module[i].forward(atom_x, pair_x,
                                                      atom_only=True)
            else:
                # not last layer, both `atom_x` and `pair_x` are needed
                atom_x, pair_x = self.weave_module[i].forward(atom_x, pair_x)
        x = readout(atom_x, mode=self.readout_mode, axis=1)
        return x    
    
if __name__ == '__main__':
    pass
