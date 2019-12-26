import torch
from torch.nn import Parameter
import torch.nn.functional as F

from . import GNNConv
from ..init import feedforward_init


class CensNetConv(GNNConv):
    def __init__(self,
                 node_in_channels, edge_in_channels,
                 node_out_channels, edge_out_channels,
                 max_n_edges=100, bias=True, **kwargs):
        '''
        ref) https://www.ijcai.org/proceedings/2019/0369.pdf
        '''
        super(CensNetConv, self).__init__()
        self.node_in_node_weight = Parameter(torch.Tensor(node_in_channels, node_out_channels))
        self.edge_in_node_weight = Parameter(torch.Tensor(edge_in_channels))

        self.node_in_edge_weight = Parameter(torch.Tensor(node_in_channels))
        self.edge_in_edge_weight = Parameter(torch.Tensor(edge_in_channels, edge_out_channels))
        if bias:
            self.node_in_node_bias = Parameter(torch.Tensor(node_out_channels))
            self.edge_in_edge_bias = Parameter(torch.Tensor(edge_out_channels))
        else:
            self.register_parameter('node_in_node_bias', None)
            self.register_parameter('edge_in_edge_bias', None)
        self.max_n_edges = max_n_edges
        self.edge_adj = None
        self.reset_parameters()

    def reset_parameters(self):
        feedforward_init(self)

    def _create_edge_adj(self, edges):
        '''
        Edge numbering method is based on Row-First.

        >>>> adj = [[0 1 0], [1 0 0], [0 1 0]]
        1st edge = adj[0, 1]
        2nd edge = adj[1, 0]
        3rd egge = adj[2, 1]

        # todo: if graph is undirected, use torch.triu(torch.ones(hoge, hoge)) to decrease computaional time.

        '''
        B = edges.size(0)
        edge_adj = torch.zeros(B, self.max_n_edges, self.max_n_edges)
        edge_list = (edges == 1).nonzero()
        n_edges = edge_list.size(0)
        edge_idx = 0

        for idx in range(n_edges - 1):
            if edge_list[idx, 0] != edge_list[idx+1, 0]:
                edge_idx = 0
            B_i = edge_list[idx, 0]
            tmp_edge_idx = edge_idx + 1
            for jdx in range(idx+1, n_edges):
                B_j = edge_list[jdx, 0]
                if B_i != B_j:
                    break
                if (edge_list[idx, 1] == edge_list[jdx, 1]) or \
                   (edge_list[idx, 2] == edge_list[jdx, 2]):
                    edge_adj[B_i, edge_idx, tmp_edge_idx] = 1
                tmp_edge_idx += 1
            edge_idx += 1
        return edge_adj

    def _create_binary_transformation_matrix(self, edges):
        if self.edge_adj is not None:
            return self.edge_adj
        B, n_nodes = edges.size(0), edges.size(1)
        binary_transformation_matrix = torch.zeros(B, n_nodes, self.max_n_edges)
        edge_list = (edges == 1).nonzero()

        edge_idx = 0
        _B = 0
        for idx, edge in enumerate(edge_list):
            B, n1, n2 = edge[0], edge[1], edge[2]
            if _B != B:
                _B = B
                edge_idx = 0
            binary_transformation_matrix[B, n1, edge_idx] = 1
            binary_transformation_matrix[B, n2, edge_idx] = 1
            edge_idx += 1
        return binary_transformation_matrix

    def node_forward(self, nodes, edges, edge_features, edge_adj):
        h_node = nodes.matmul(self.node_in_node_weight)
        if self.node_in_node_bias is not None:
             h_node = h_node + self.node_in_node_bias
        adj = self._laplacianize(edges)
        h_node = adj.matmul(h_node)
        h_edge = edge_features.matmul(self.edge_in_node_weight)

        btm = self._create_binary_transformation_matrix(edges)
        diag_edge = torch.diag_embed(h_edge, dim1=1, dim2=2)
        h_edge = btm.matmul(diag_edge)
        h_edge = h_edge.matmul(btm.transpose(1, 2))
        return h_edge * h_node

    def edge_forward(self, nodes, edges, edge_features, edge_adj):
        h_edge = edge_features.matmul(self.edge_in_edge_weight)
        if self.edge_in_edge_bias is not None:
             h_edge = h_edge + self.edge_in_edge_bias
        edge_adj = self._laplacianize(edge_adj)
        h_edge = edge_adj.matmul(h_edge)

        btm = self._create_binary_transformation_matrix(edges)
        h_node = nodes.matmul(self.node_in_edge_weight)
        diag_node = torch.diag_embed(h_node, dim1=1, dim2=2)
        h_node = btm.transpose(1, 2).matmul(diag_node.matmul(btm))
        return h_node * h_edge

    def forward(self, nodes, edges, edge_features):
        edge_adj = self._create_edge_adj(edges)
        h_node = self.node_forward(nodes, edges, edge_features, edge_adj)
        h_edge = self.edge_forward(nodes, edges, edge_features, edge_adj)
        return h_node, h_edge

if __name__ == '__main__':
    c = CensNetConv(8, 4, 10, 100, 100)
    torch.manual_seed(0)
    edges = torch.empty(2, 10, 10).random_(2)
    nodes = torch.randn(2, 10, 8)
    edge_features = torch.randn(2, 100, 4)
    hn, he = c(nodes, edges, edge_features)
    c = CensNetConv(10, 100, 12, 100, 100)
    hn, he = c(hn, edges, he)
    print(hn.shape, he.shape)
