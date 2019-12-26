import torch


def batched_sparse_eyes(node_size: tuple or list, batch_size: int, dtype, device):
    i = torch.arange(node_size).repeat(batch_size)
    batch_indceis = torch.repeat_interleave(torch.arange(batch_size), node_size)
    i = torch.stack([batch_indceis, i, i])
    v = torch.ones(node_size * batch_size).long()
    sparsed_identity = torch.sparse_coo_tensor(i, v, [batch_size, node_size, node_size],
                                               dtype=dtype, device=device)
    return sparsed_identity
