import tensorly as tl

def matrix_product_state(input_tensor, rank = 10, verbose = False):
    """MPS decomposition via recursive SVD
        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)

    Parameters
    ----------
    input_tensor : ndarray
    rank : {int, int list}, optional
                  maximum allowable MPS rank of the factors
                  if int, then this is the same for all the factors
                  if int list, then the kth entry in the list is the maximum
                  allowable MPS rank for the kth factor
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : MPS factors
              order-3 tensors of the MPS decomposition

    References
    ----------
    .. [1] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing,
       33(5):2295–2317, 2011.
    """

    # Check user input for errors
    tensor_dimensions = len(input_tensor.shape)
    if isinstance(rank, int) is True:
        rank = [rank] * (tensor_dimensions+1)

    else:
        error_message = "Incorrect size of rank array "
        error_message += str(tensor_dimensions+1) + " != " + str(len(rank))
        assert(tensor_dimensions+1 == len(rank)), error_message

    # Get basic information on of input tensor
    n_mode_dimensions = input_tensor.shape
    D = len(n_mode_dimensions)
    context = tl.context(input_tensor)

    # Initialization
    unfolding_matrix = input_tensor
    mps_ranks = [None] * (D+1)
    mps_ranks[0] = 1
    mps_ranks[D] = 1
    factors = [None] * D

    # Getting the MPS factors up to D-1
    for k in range(D-1):

        # Reshape the unfolding matrix of the remaining factors
        num_rows = mps_ranks[k] * int(n_mode_dimensions[k])
        unfolding_matrix = tl.reshape(unfolding_matrix, (num_rows, -1))

        # SVD of unfolding matrix
        (num_rows, num_columns) = unfolding_matrix.shape
        mps_rank = min(num_rows, num_columns, rank[k+1])
        U, s, V_t = tl.partial_svd(unfolding_matrix, mps_rank)
        mps_ranks[k+1] = mps_rank

        # Get kth MPS factor
        factors[k] = tl.reshape(U, (mps_ranks[k], n_mode_dimensions[k], mps_ranks[k+1]))

        if(verbose is True):
            print("MPS factor " + str(k) + " computed with shape " + str(factors[k].shape))

        # Get new unfolding matrix for the remaining factors
        (rows, cols) = V_t.shape
        unfolding_matrix = tl.zeros((rows, cols), **context)

        for i in range(rows):
            unfolding_matrix[i, :] = s[i] * V_t[i, :]

    # Getting the last factor
    (r_prev_D, n_D) = unfolding_matrix.shape
    factors[D-1] = tl.reshape(unfolding_matrix, (r_prev_D, n_D, 1))

    if(verbose is True):
        print("MPS factor " + str(D-1) + " computed with shape " + str(factors[D-1].shape))

    return factors
