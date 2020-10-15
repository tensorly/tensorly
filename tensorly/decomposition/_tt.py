import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..tt_tensor import _validate_tt_rank

def matrix_product_state(input_tensor, rank, verbose=False):
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
    rank = _validate_tt_rank(tl.shape(input_tensor), rank=rank)
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

    return factors


class TensorTrain(DecompositionMixin):
    def __init__(self, rank, verbose=False):
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
        self.rank = rank
        self.verbose = verbose

    def fit_transform(self, tensor):
        self.decomposition_ = matrix_product_state(tensor, rank=self.rank, verbose=self.verbose)
        return self.decomposition_
