import tensorly as tl
from ..tr_tensor import validate_tr_rank, TRTensor


def tensor_ring(input_tensor, rank, mode=0, verbose=False):
    """Tensor Ring decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors) [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : Union[int, List[int]]
            maximum allowable TR rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    mode : int, default is 0
            index of the first factor to compute
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TR factors
              order-3 tensors of the TR decomposition

    References
    ----------
    .. [1] Qibin Zhao et al. "Tensor Ring Decomposition" arXiv preprint arXiv:1606.05535, (2016).
    """
    rank = validate_tr_rank(tl.shape(input_tensor), rank=rank)
    n_dim = len(input_tensor.shape)

    # Change order
    if mode:
        order = tuple(range(mode, n_dim)) + tuple(range(mode))
        input_tensor = tl.transpose(input_tensor, order)
        rank = rank[mode:] + rank[:mode]

    tensor_size = input_tensor.shape

    factors = [None] * n_dim

    # Getting the first factor
    unfolding = tl.reshape(input_tensor, (tensor_size[0], -1))

    n_row, n_column = unfolding.shape
    if rank[0] * rank[1] > min(n_row, n_column):
        raise ValueError(f'rank[{mode}] * rank[{mode + 1}] = {rank[0] * rank[1]} is larger than '
                         f'first matricization dimension {n_row}×{n_column}.\n'
                         'Failed to compute first factor with specified rank. '
                         'Reduce specified ranks or change first matricization `mode`.')

    # SVD of unfolding matrix
    U, S, V = tl.partial_svd(unfolding, rank[0] * rank[1])

    # Get first TR factor
    factor = tl.reshape(U, (tensor_size[0], rank[0], rank[1]))
    factors[0] = tl.transpose(factor, (1, 0, 2))
    if verbose is True:
        print("TR factor " + str(mode) + " computed with shape " + str(factor.shape))

    # Get new unfolding matrix for the remaining factors
    unfolding = tl.reshape(S, (-1, 1)) * V
    unfolding = tl.reshape(unfolding, (rank[0], rank[1], -1))
    unfolding = tl.transpose(unfolding, (1, 2, 0))

    # Getting the TR factors up to n_dim - 1
    for k in range(1, n_dim - 1):

        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        n_row, n_column = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        U, S, V = tl.partial_svd(unfolding, current_rank)
        rank[k + 1] = current_rank

        # Get kth TR factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))

        if verbose is True:
            print("TR factor " + str((mode + k) % n_dim) + " computed with shape " + str(factors[k].shape))

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    prev_rank = unfolding.shape[0]
    factors[-1] = tl.reshape(unfolding, (prev_rank, -1, rank[0]))

    if verbose is True:
        print("TR factor " + str((mode - 1) % n_dim) + " computed with shape " + str(factors[-1].shape))

    # Reorder factors to match input
    if mode:
        factors = factors[-mode:] + factors[:-mode]

    return TRTensor(factors)
