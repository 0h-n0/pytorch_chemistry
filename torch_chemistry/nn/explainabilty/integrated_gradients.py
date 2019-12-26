import torch
import torch.nn as nn

from . import Explainabilty


class IntegratedGradients(Explainabilty):
    def __init__(self,
                 model: nn.Module,
                 divided_number: int = 100,
                 grad_target_is_label: bool = False,
                 edge_scale: bool = False):
        '''
        ref) https://arxiv.org/abs/1703.01365
        '''
        self.model = model
        self.divided_number = divided_number
        self.grad_target_is_label = grad_target_is_label
        self.edge_scale = edge_scale

    def __call__(self, nodes, edges, labels,
                 nodes_base=None, edges_base=None,
                 *args, **kwargs):
        '''
        '''
        self.model.eval()
        B, _, _ = nodes.shape
        ig = torch.zeros_like(nodes)

        if nodes_base is None:
            nodes_base = torch.zeros_like(nodes)

        if edges_base is None:
            edges_base = torch.zeros_like(edges)

        if not nodes.requires_grad:
            nodes.requires_grad = True

        if not edges.requires_grad:
            edges.requires_grad = True

        for k in range(self.divided_number):
            self.model.zero_grad()
            alpha = (k + 1) / float(self.divided_number)
            _nodes = nodes_base + (nodes - nodes_base) * alpha
            if self.edge_scale:
                _edges = edge_base + (edges - edges_base) * alpha
            else:
                _edges = edges
            output = self.model(_nodes, _edges, *args, **kwargs)
            labels = labels.reshape(-1).long()
            l = output[:, labels]
            ig = torch.autograd.grad(_nodes, l, o)



class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.g = GCNConv(8, 16)
        self.l = nn.Linear(160, 10)
    def forward(self, nodes, edges):
        o = self.g(nodes, edges)
        o = o.reshape(o.size(0), -1)
        o = self.l(o)
        return o

if __name__ == '__main__':
    from ..conv.gcn_conv import GCNConv
    nodes = torch.randn(10, 10, 8)
    edges = torch.randn(10, 10, 10).random_(0, 2)
    labels = torch.randn(10, 1).random_(0, 2)
    g = GCN()
    explainabilty = IntegratedGradients(g)
    explainabilty(nodes, edges, labels)
