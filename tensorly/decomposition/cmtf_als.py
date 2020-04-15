import numpy as np

import tensorly as tl
from ..tenalg import khatri_rao
from ..random import random_kruskal
from ..tenalg import solve_least_squares
from ..kruskal_tensor import kruskal_normalise
from ..kruskal_tensor import KruskalTensor


# Authors: Isabell Lehmann <isabell.lehmann94@outlook.de>

# License: BSD 3 clause


def align_tensors(permutation_matrix):
    """
    The order of the rows in the matrix is adjusted such that the highest values are on the main
    diagonal.

    :param permutation_matrix: pre-calculated permutation matrix, values can be in the range [-1, 1]
    :return order: order of the rows such that the highest values are on the main diagonal
                    -> matrix = matrix[order, :]
    """

    order = tl.argmax(permutation_matrix, axis=0)
    uniq, uniq_idx, counts = np.unique(order, return_index=True, return_counts=True)
    if len(uniq) < len(order):
        non_uniq = uniq[counts > 1][0]
        # idea works as long as there are only two equal values in order (non_uniq_idx has length 2)
        non_uniq_idx = np.where(order == non_uniq)[0]
        wrong_idx = non_uniq_idx[np.argmin(permutation_matrix[np.full_like(non_uniq_idx, non_uniq),
                                                              non_uniq_idx])]
        order[wrong_idx] = np.setdiff1d(np.arange(len(order)), order)[0]

    return order


def factor_match_score_3d(kruskal_tensor_3d_true, kruskal_tensor_2d_true, kruskal_tensor_3d_pred,
                          kruskal_tensor_2d_pred):
    """
    Calculates factor match score (FMS) for comparing true and estimated matrices.

    Assume a 3rd order tensor and a matrix are coupled in the 1th mode with
    X = [[A, B, C]] and Y = [[A, V]]
    The FMS shows how close the estimation of A, B, C, V is to the true matrices.
    An FMS of 1 is a perfect match. The worst value is 0.

    Formula for FMS according to Equation 4
    @article{acar2011all,
      title={All-at-once optimization for coupled matrix and tensor factorizations},
      author={Acar, Evrim and Kolda, Tamara G and Dunlavy, Daniel M},
      journal={arXiv preprint arXiv:1105.3422},
      year={2011}
    }

    Notes
    ------
    In a first step, the tensors are aligned. Therefore, <a_i, a_pred_j> * <b_i, b_pred_j> *
    <c_i, c_pred_j> is calculated for all i and j, not only for i=j. If the i-th vector of the
    true factor matrix belongs to the j-th vector of the predicted matrix, the result is close to
    1, otherwise very small. The order of the predicted factors is adjusted in the align_tensors
    function, so that the i-th true factor belongs to the i-th predicted factor, i.e. that
    (A.T @ A_pred) * (B.T @ B_pred) * (C.T @ C_pred) consists of values close to 1 on its main
    diagonal.

    Parameters
    ----------
    kruskal_tensor_3d_true : Kruskal tensor
        true tensor X = [[A, B, C]]
    kruskal_tensor_2d_true : Kruskal tensor
        true matrix Y = [[A, V]]
    kruskal_tensor_3d_pred : Kruskal tensor
        tensor consisting of predicted factor matrices A_pred, B_pred, C_pred: X_pred = [[A_pred,
        B_pred, C_pred]]
    kruskal_tensor_2d_pred : Kruskal tensor
        matrix consisting of predicted factor matrices A_pred, V_pred: Y_pred = [[A_pred, V_pred]]

    Returns
    -------
    factor match score : float
        value between 0 and 1 that shows how good the estimation is, 1 is perfect

    """

    lambda_, [a, B, C] = kruskal_normalise(kruskal_tensor_3d_true)
    alpha_, [A, V] = kruskal_normalise(kruskal_tensor_2d_true)
    xi = lambda_ + alpha_

    lambda_pred, [a_pred, B_pred, C_pred] = kruskal_normalise(kruskal_tensor_3d_pred)
    alpha_pred, [A_pred, V_pred] = kruskal_normalise(kruskal_tensor_2d_pred)
    xi_pred = lambda_pred + alpha_pred

    # take care of permutation of normalized factor matrices and of sign of weights (not mentioned
    # in paper, but necessary)
    product = tl.dot(tl.transpose(A), A_pred) * tl.dot(tl.transpose(B), B_pred) * tl.dot(
        tl.transpose(C), C_pred)
    perm = product * np.sign(lambda_[:, None] @ lambda_pred[None, :])
    order = align_tensors(perm)

    product = product[order, :]  # product[r,r] now corresponds to
    # (a_r.T @ a_pred_r) / (||(a_r|| ||a_pred_r||) * (b_r.T @ b_pred_r) / (||(b_r|| ||b_pred_r||) *
    # (c_r.T @ c_pred_r) / (||(c_r|| ||c_pred_r||)
    V = V[:, order]
    xi = xi[order]

    # calculate factor match score
    rank = A.shape[1]
    FMS = tl.zeros(rank)
    for r in range(rank):
        FMS[r] = (1 - tl.abs(xi[r] - xi_pred[r]) / tl.max([xi[r], xi_pred[r]])) * tl.abs(
            product[r, r] * tl.dot(V[:, r], V_pred[:, r]))

    return tl.min(FMS)


def coupled_matrix_tensor_3d_factorization(tensor_3d, matrix, rank):
    """
    Calculates a coupled matrix and tensor factorization of 3rd order tensor and matrix which are coupled in first mode.

    Assume you have tensor_3d = [[A, B, C]] and matrix = [[A, V]], which are coupled in 1st mode.
    With coupled matrix and tensor factorization (CTMF), the factor matrices A, B, C for the CP decomposition of X
    and the matrix V are found.
    This implementation only works for a coupling in the first mode.

    Solution is found via alternating least squares (ALS) as described in Figure 5 of
    @article{acar2011all,
      title={All-at-once optimization for coupled matrix and tensor factorizations},
      author={Acar, Evrim and Kolda, Tamara G and Dunlavy, Daniel M},
      journal={arXiv preprint arXiv:1105.3422},
      year={2011}
    }

    Parameters
    ----------
    tensor_3d : tl.tensor or Kruskal tensor
        3rd order tensor X = [[A, B, C]]
    matrix : tl.tensor or Kruskal tensor
        matrix that is coupled with tensor in first mode: Y = [[A, V]]
    rank : int
        rank for CP decomposition of X

    Returns
    -------
    tensor_3d_pred, matrix_pred : Kruskal tensors
        tensor_3d_pred = [[A,B,C]], matrix_pred = [[A,V]]

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

    if tl.is_tensor(tensor_3d):
        X = tensor_3d
    else:
        shape, _ = tl.kruskal_tensor._validate_kruskal_tensor(
            tensor_3d)  # this will fail if it isn't a valid tuple or KruskalTensor
        X = tl.kruskal_tensor.kruskal_to_tensor(tensor_3d)

    if tl.is_tensor(matrix):
        Y = matrix
    else:
        shape, _ = tl.kruskal_tensor._validate_kruskal_tensor(
            matrix)  # this will fail if it isn't a valid tuple or KruskalTensor
        Y = tl.kruskal_tensor.kruskal_to_tensor(matrix)

    # initialize values
    s = X.shape + (Y.shape[1],)
    A, B, C, V = random_kruskal(s, rank).factors

    # alternating least squares
    # note that no rescaling is done since it is not guaranteed that the columns in true matrices
    # have unit norm
    # note that the order of the khatri rao product is reversed since tl.unfold has another order
    # than assumed in paper
    for iteration in range(10 ** 4):
        A = tl.transpose(solve_least_squares(
            tl.transpose(tl.concatenate((tl.transpose(khatri_rao([B, C])), tl.transpose(V)),
                                        axis=1)),
            tl.transpose(tl.concatenate((tl.unfold(X, 0), Y), axis=1))))
        B = tl.transpose(solve_least_squares(khatri_rao([A, C]), tl.transpose(tl.unfold(X, 1))))
        C = tl.transpose(solve_least_squares(khatri_rao([A, B]), tl.transpose(tl.unfold(X, 2))))
        V = tl.transpose(solve_least_squares(A, Y))
        error_new = 1 / 2 * tl.norm(
            X - tl.kruskal_tensor.kruskal_to_tensor((None, [A, B, C]))) ** 2 + 1 / 2 * tl.norm(
            Y - tl.kruskal_tensor.kruskal_to_tensor((None, [A, V])))

        if iteration > 0 and tl.abs(error_new - error_old) / error_old <= 1e-8:
            break
        error_old = error_new

    tensor_3d_pred = KruskalTensor((None, [A, B, C]))
    matrix_pred = KruskalTensor((None, [A, V]))

    return tensor_3d_pred, matrix_pred
