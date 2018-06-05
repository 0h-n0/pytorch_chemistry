from pytorch_chemistry.models.functions import GraphMaxPool

def graph_max_pooling(x, pooling_inds):
    """Graph max pooling function.
    This function computes the maximum of the values at a specified
    set of vertices.
    Args:
        x (~torch.autograd.Variable): Input variable.
        pooling_inds: (~ndarray): 2D array that specifies the indices of the
            vertices that should be max pooled. The first dimension is equal
            to the number of vertices in the resulting pooled graph.
            The second dimension is equal to the number of vertices to pool
            in order to produce the corresponding vertex in the pooled graph.
    Returns:
        ~torch.autograd.Variable: Output variable.
    """
    return GraphMaxPool(pooling_inds)(x)
