import torch
from torch.nn.parameter import Parameter

from e2edd import MAX_ATOMIC_NUM
from pytorch_chemistry.models.basic import init_feedforward_weights

class EmbedAtomID(torch.nn.Module):
    """Embeddning specialized to atoms.

    This is a chain in the sense of Pytorch that converts
    an atom, represented by a sequence of molecule IDs,
    to a sequence of embedding vectors of molecules.
    The operation is done in a minibatch manner, as most chains do.

    The forward propagation of link consists of ID embedding,
    which converts the input `x` into vector embedding `h` where
    its shape represents (minibatch, atom, channel)

    .. seealso:: :class:`torch.nn.Embedding`
    """
    def __init__(self,
                 embedding_dim,   # out_dim
                 num_embeddings=MAX_ATOMIC_NUM,                 
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False):
        super(EmbedAtomID, self).__init__()
        self.embedding = torch.nn.Embedding(
                 num_embeddings=num_embeddings,            
                 embedding_dim=embedding_dim,
                 padding_idx=padding_idx,
                 max_norm=max_norm,
                 norm_type=norm_type,
                 scale_grad_by_freq=scale_grad_by_freq,
                 sparse=sparse)

    def __call__(self, x):
        """Forward propagaion.

        Args:
            x (:class:`torch.autograd.Variable`, or :class:`numpy.ndarray` \
            or :class:`cupy.ndarray`):
                Input array that should be an integer array
                whose ``ndim`` is 2. This method treats the array
                as a minibatch of atoms, each of which consists
                of a sequence of molecules represented by integer IDs.
                The first axis should be an index of atoms
                (i.e. minibatch dimension) and the second one be an
                index of molecules.

        Returns:
            :class:`torch.autograd.Variable`:
                A 3-dimensional array consisting of embedded vectors of atoms,
                representing (minibatch, atom, channel).

        """
        return self.embedding(x)

    
class GraphLinear(torch.nn.Module):
    """Graph Linear layer.

    This function assumes its input is 3-dimensional.
    Differently from :class:`chainer.functions.linear`, it applies an affine
    transformation to the third axis of input `x`.

    .. seealso:: :class:`torch.nn.Linear`
    """
    def __init__(self,
                 in_features,
                 out_features,
                 *,
                 nonlinearity='sigmoid',
                 init_mean=0,
                 init_std=1,
                 init_xavier: bool=True,
                 init_normal: bool=True,
                 init_gain = None,
                 dropout=0.0,
                 bias=True,
    ):

        super(GraphLinear, self).__init__()

        self.linear = torch.nn.Linear(in_features,
                                      out_features,
                                      bias)
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        
        if not init_gain and nonlinearity is not None:
            init_gain = torch.nn.init.calculate_gain(nonlinearity)
        else:
            init_gain = 1
        
        init_feedforward_weights(self.linear,
                                 init_mean,
                                 init_std,
                                 init_xavier,
                                 init_normal,
                                 init_gain)

    def __call__(self, x):
        """Forward propagation.

        Args:
            x (:class:`chainer.Variable`, or :class:`numpy.ndarray`\
            or :class:`cupy.ndarray`):
                Input array that should be a float array whose ``ndim`` is 3.

                It represents a minibatch of atoms, each of which consists
                of a sequence of molecules. Each molecule is represented
                by integer IDs. The first axis is an index of atoms
                (i.e. minibatch dimension) and the second one an index
                of molecules.

        Returns:
            :class:`chainer.Variable`:
                A 3-dimeisional array.

        """
        # (minibatch, atom, ch)
        s0, s1, s2 = x.size()
        x = x.view(s0 * s1, s2)
        x = self.linear(x)
        x = x.view(s0, s1, self.out_features)
        return x

class GraphBatchNorm(torch.nn.Module):
    """Graph Batch Normalization layer.

    .. seealso:: :class:`torch.autograd.BatchNorm1d`
    """
    def __init__(self, num_features):
        super(GraphBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)

    def __call__(self, x):
        """Forward propagation.

        Args:
            x (:class:`torch.autograd.Variable`, or :class:`numpy.ndarray`
                Input array that should be a float array whose ``ndim`` is 3.

                It represents a minibatch of atoms, each of which consists
                of a sequence of molecules. Each molecule is represented
                by integer IDs. The first axis is an index of atoms
                (i.e. minibatch dimension) and the second one an index
                of molecules.

        Returns:
            :class:`touch.autograd.Variable`:
                A 3-dimeisional array.

        """
        # (minibatch, atom, ch)

        # The implemenataion of batch normalization for graph convolution below
        # is rather naive. To be precise, it is necessary to consider the
        # difference in the number of atoms for each graph. However, the
        # implementation below does not take it into account, and assumes
        # that all graphs have the same number of atoms, hence extra numbers
        # of zero are included when average is computed. In other word, the
        # results of batch normalization below is biased.

        s0, s1, s2 = x.size()
        x = x.view(s0 * s1, s2)
        x = self.bn(x)
        x = x.view(s0, s1, s2)

        return x


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        init_feedforward_weights(self)
        
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'        
        
class GraphMaxPool(torch.nn.Module):
    pass


if __name__ == '__main__':
    embedding = EmbedAtomID(10)
    input = torch.autograd.Variable(torch.LongTensor([[1,2,4,5]]))
    print(embedding)
    embedding(input)
            
    g = GraphLinear(3, 10)
    t = torch.ones(3, 10, 3)
    input = torch.autograd.Variable(t)
    g(input)

    b = GraphBatchNorm(3)
    t = torch.ones(3, 10, 3)
    input = torch.autograd.Variable(t)
    b(input)

    g = GraphConvolution(3, 10)
    print(g)
