import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..tt_tensor import validate_tt_rank, TTTensor
from ..tt_matrix import validate_tt_matrix_rank, TTMatrix
from ..utils import DefineDeprecated

def tensor_train(input_tensor, rank, verbose=False):
    """TT decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)
        -- also known as Tensor-Train decomposition [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable TT rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TT factors
              order-3 tensors of the TT decomposition

    References
    ----------
    .. [1] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing, 33(5):2295–2317, 2011.
    """
    rank = validate_tt_rank(tl.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    unfolding = input_tensor
    factors = [None] * n_dim

    # Getting the TT factors up to n_dim - 1
    for k in range(n_dim - 1):

        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k]*tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k+1])
        U, S, V = tl.partial_svd(unfolding, current_rank)
        rank[k+1] = current_rank

        # Get kth TT factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k+1]))

        if(verbose is True):
            print("TT factor " + str(k) + " computed with shape " + str(factors[k].shape))

        # Get new unfolding matrix for the remaining factors
        unfolding= tl.reshape(S, (-1, 1))*V

    # Getting the last factor
    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = tl.reshape(unfolding, (prev_rank, last_dim, 1))

    if(verbose is True):
        print("TT factor " + str(n_dim-1) + " computed with shape " + str(factors[n_dim-1].shape))

    return TTTensor(factors)

def tensor_train_matrix(tensor, rank):
    """Decompose a tensor into a matrix in tt-format
    
    Parameters
    ----------
    tensor : tensorized matrix 
        if your input matrix is of size (4, 9) and your tensorized_shape (2, 2, 3, 3)
        then tensor should be tl.reshape(matrix, (2, 2, 3, 3))
    rank : 'same', float or int tuple
        - if 'same' creates a decomposition with the same number of parameters as `tensor`
        - if float, creates a decomposition with `rank` x the number of parameters of `tensor`
        - otherwise, the actual rank to be used, e.g. (1, rank_2, ..., 1) of size tensor.ndim//2. Note that boundary conditions dictate that the first rank = last rank = 1.
    
    Returns
    -------
    tt_matrix
    """
    order = tl.ndim(tensor)
    n_input = order // 2 # (n_output = n_input)
    
    if tl.ndim(tensor) != n_input*2:
        msg = 'The tensor should have as many dimensions for inputs and outputs, i.e. order should be even '
        msg += f'but got a tensor of order tl.ndim(tensor)={order} which is odd.'
        raise ValueError(msg)

    in_shape = tl.shape(tensor)[:n_input]
    out_shape = tl.shape(tensor)[n_input:]

    if n_input == 1:
        # A TTM with a single factor is just a matrix...
        return TTMatrix([tensor.reshape(1, in_shape[0], out_shape[0], 1)])

    new_idx = list([idx for tuple_ in zip(range(n_input), range(n_input, 2*n_input)) for idx in tuple_])
    new_shape = list([a*b for (a,b) in zip(in_shape, out_shape)])
    tensor = tl.reshape(tl.transpose(tensor, new_idx), new_shape)
    
    factors = tensor_train(tensor, rank).factors
    for i in range(len(factors)):
        factors[i] = tl.reshape(factors[i], (factors[i].shape[0], in_shape[i], out_shape[i], -1))
    
    return TTMatrix(factors)


class TensorTrain(DecompositionMixin):
    """TT decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)
        -- also known as Tensor-Train decomposition [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable TT rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TT factors
            order-3 tensors of the TT decomposition

    References
    ----------
    .. [1] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing, 33(5):2295–2317, 2011.
    """
    def __init__(self, rank, verbose=False):
        self.rank = rank
        self.verbose = verbose

    def fit_transform(self, tensor):
        self.decomposition_ = tensor_train(tensor, rank=self.rank, verbose=self.verbose)
        return self.decomposition_

matrix_product_state = DefineDeprecated('matrix_product_state', tensor_train)
