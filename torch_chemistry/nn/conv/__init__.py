import torch
import torch.nn as nn


class GNNConv(nn.Module):
    def __init__(self):
        super(GNNConv, self).__init__()

    def _laplacianize(self, adj):
        return self._normalize(self._add_identity(adj))

    def _add_identity(self, adj):
        B, N, _ = adj.size()
        eyes = torch.eye(N, dtype=adj.dtype, device=adj.device).reshape(-1, N, N).repeat(B, 1, 1)
        return adj.add(eyes)

    def _normalize(self, adj):
        inv_sqrt_degree_mat = 1 / torch.sqrt(torch.diag_embed(adj.sum(axis=2)))
        inv_sqrt_degree_mat[inv_sqrt_degree_mat == float('inf')] = 0
        adj = adj.matmul(inv_sqrt_degree_mat)
        normalized_adj = inv_sqrt_degree_mat.matmul(adj)
        return normalized_adj
