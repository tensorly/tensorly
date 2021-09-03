import tensorly as tl
from ..tenalg import khatri_rao
from ..cp_tensor import CPTensor, validate_cp_rank, cp_to_tensor, cp_normalize
from ._cp import initialize_cp


# Authors: Isabell Lehmann <isabell.lehmann94@outlook.de>

# License: BSD 3 clause



def coupled_matrix_tensor_3d_factorization(tensor_3d, matrix, rank, init='svd', n_iter_max=100, normalize_factors=False):
    """
    Calculates a coupled matrix and tensor factorization of 3rd order tensor and matrix which are
    coupled in first mode.

    Assume you have tensor_3d = [[lambda; A, B, C]] and matrix = [[gamma; A, V]], which are
    coupled in 1st mode. With coupled matrix and tensor factorization (CTMF), the normalized
    factor matrices A, B, C for the CP decomposition of X, the normalized matrix V and the
    weights lambda_ and gamma are found. This implementation only works for a coupling in the
    first mode.

    Solution is found via alternating least squares (ALS) as described in Figure 5 of
    @article{acar2011all,
      title={All-at-once optimization for coupled matrix and tensor factorizations},
      author={Acar, Evrim and Kolda, Tamara G and Dunlavy, Daniel M},
      journal={arXiv preprint arXiv:1105.3422},
      year={2011}
    }

    Notes
    -----
    In the paper, the columns of the factor matrices are not normalized and therefore weights are
    not included in the algorithm.

    Parameters
    ----------
    tensor_3d : tl.tensor or CP tensor
        3rd order tensor X = [[A, B, C]]
    matrix : tl.tensor or CP tensor
        matrix that is coupled with tensor in first mode: Y = [[A, V]]
    rank : int
        rank for CP decomposition of X

    Returns
    -------
    tensor_3d_pred : CPTensor
        tensor_3d_pred = [[lambda; A,B,C]]
    matrix_pred : CPTensor
        matrix_pred = [[gamma; A,V]]
    rec_errors : list
        contains the reconstruction error of each iteration:
        error = 1 / 2 * | X - [[ lambda_; A, B, C ]] | ^ 2 + 1 / 2 * | Y - [[ gamma; A, V ]] | ^ 2

    Examples
    --------
    A = tl.tensor([[1, 2], [3, 4]])
    B = tl.tensor([[1, 0], [0, 2]])
    C = tl.tensor([[2, 0], [0, 1]])
    V = tl.tensor([[2, 0], [0, 1]])
    R = 2

    X = (None, [A, B, C])
    Y = (None, [A, V])

    tensor_3d_pred, matrix_pred = cmtf_als_for_third_order_tensor(X, Y, R)

    """
    rank = validate_cp_rank(tl.shape(tensor_3d), rank=rank)

    # initialize values
    tensor_cp = initialize_cp(tensor_3d, rank, init=init)
    rec_errors = []

    # alternating least squares
    # note that the order of the khatri rao product is reversed since tl.unfold has another order
    # than assumed in paper
    for iteration in range(n_iter_max):
        V = tl.transpose(tl.lstsq(tensor_cp.factors[0], matrix)[0])

        # Loop over modes of the tensor
        for ii in range(tl.ndim(tensor_3d)):
            kr = khatri_rao(tensor_cp.factors, skip_matrix=ii)
            unfolded = tl.unfold(tensor_3d, ii)

            # If we are at the coupled mode, concat the matrix
            if ii == 0:
                kr = tl.concatenate((kr, V), axis=0)
                unfolded = tl.concatenate((unfolded, matrix), axis=1)

            tensor_cp.factors[ii] = tl.transpose(tl.lstsq(kr, tl.transpose(unfolded))[0])

        error_new = tl.norm(
            tensor_3d - cp_to_tensor(tensor_cp)) ** 2 + tl.norm(
            matrix - cp_to_tensor((None, [tensor_cp.factors[0], V]))) ** 2

        if iteration > 0 and (tl.abs(error_new - error_old) / error_old <= 1e-8 or error_new <
                              1e-5):
            break
        error_old = error_new
        rec_errors.append(error_new)

    matrix_pred = CPTensor((None, [tensor_cp.factors[0], V]))

    if normalize_factors:
        tensor_cp = cp_normalize(tensor_cp)
        matrix_pred = cp_normalize(matrix_pred)

    return tensor_cp, matrix_pred, rec_errors
