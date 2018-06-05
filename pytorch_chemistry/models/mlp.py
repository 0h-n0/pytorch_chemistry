import torch.nn as nn


class MLP(nn.Module):
    """Basic implementation for MLP

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        activation (chainer.functions): activation function
    """

    def __init__(self, input_dim, out_dim, hidden_dim=16,
                 n_layers=2, activation='relu'):
        super(MLP, self).__init__()
        assert n_layers >= 0, 'ValueError'
        if n_layers == 1:
            self.nn = nn.Linear(input_dim, hidden_dim)
        elif n_layers == 2:
            self.nn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                )
        else:
            _module_list = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for i in range(n_layers - 1):
                _module_list.append(nn.Linear(hidden_dim, hidden_dim))
                _module_list.append(nn.ReLU())
            _module_list.append(nn.Linear(hidden_dim, out_dim))                
            self.nn = nn.ModuleList(_module_list)

    def forward(self, x):
        for i, l in enumerate(self.nn):
            x = l(x)
        return x
        

if __name__ == '__main__':
    m = MLP(10, 3, 5, 3)
    print(m)
    m = MLP(10, 3, 5, 1)
    print(m)
    m = MLP(10, 3, 5, 2)
    print(m)
    

    

