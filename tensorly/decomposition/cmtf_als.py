import numpy as np

import tensorly as tl
from ..tenalg import khatri_rao
from ..random import random_kruskal
from ..tenalg import solve_least_squares


# Authors: Isabell Lehmann <isabell.lehmann94@outlook.de>

# License: BSD 3 clause


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

    Parameters
    ----------
    kruskal_tensor_3d_true : Kruskal tensor
        true tensor X = [[A, B, C]]
    kruskal_tensor_3d_true : Kruskal tensor
        true matrix Y = [[A, V]]
    kruskal_tensor_3d_pred : Kruskal tensor
        tensor consisting of estimated factor matrices A_pred, B_pred, C_pred: X_pred = [[A_pred, B_pred, C_pred]]
    kruskal_tensor_3d_pred : Kruskal tensor
        matrix consisting of estimated factor matrices A_pred, V_pred: Y_pred = [[A_pred, V_pred]]

    Returns
    -------
    factor match score : float
        value between 0 and 1 that shows how good the estimation is, 1 is perfect

    """

    lambda_, [a, B, C] = tl.kruskal_tensor.kruskal_normalise(kruskal_tensor_3d_true)
    alpha_, [A, V] = tl.kruskal_tensor.kruskal_normalise(kruskal_tensor_2d_true)
    xi = lambda_ + alpha_

    lambda_pred, [a_pred, B_pred, C_pred] = tl.kruskal_tensor.kruskal_normalise(kruskal_tensor_3d_pred)
    alpha_pred, [A_pred, V_pred] = tl.kruskal_tensor.kruskal_normalise(kruskal_tensor_2d_pred)
    xi_pred = lambda_pred + alpha_pred

    # take care of permutation of normalized matrices (not mentioned in paper, but necessary)
    perm = (A.T @ A_pred) * (B.T @ B_pred) * (C.T @ C_pred)
    order = tl.argmax(perm, axis=1)
    perm = perm[:, order]  # perm[r,r] now corresponds to \prod_i (a^i_r.T @ ahat^i_r) / (||(a^i_r|| ||ahat^i_r||)
    V_pred = V_pred[:, order]
    xi_hat = xi_pred[order]

    # calculate factor match score
    rank = A.shape[1]
    FMS = tl.zeros(rank)
    for r in range(rank):
        FMS[r] = (1 - tl.abs(xi[r] - xi_hat[r]) / tl.max([xi[r], xi_hat[r]])) * tl.abs(
            perm[r, r] * tl.dot(V[:, r], V_pred[:, r]))

    return tl.min(FMS)


def cmtf_als_for_third_order_tensor(tensor_3d, matrix, rank):
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

    X = tl.kruskal_tensor.KruskalTensor((None, [A, B, C]))  # weights are None
    Y = tl.kruskal_tensor.KruskalTensor((None, [A, V]))

    tensor_3d_pred, matrix_pred = cmtf_als_for_third_order_tensor(X, Y, R)

    """

    # initialize values
    shape_list = tensor_3d.shape
    s = shape_list + (matrix.shape[1],)
    A, B, C, V = random_kruskal(s, rank).factors

    if isinstance(tensor_3d, tl.kruskal_tensor.KruskalTensor):
        X = tl.kruskal_tensor.kruskal_to_tensor(tensor_3d)
    else:
        X = tensor_3d.copy()

    if isinstance(matrix, tl.kruskal_tensor.KruskalTensor):
        Y = tl.kruskal_tensor.kruskal_to_tensor(matrix)
    else:
        Y = matrix.copy()

    objective_old = tl.tensor(np.inf)

    # alternating least squares
    # note that no rescaling is done since it is not guaranteed that the columns in true matrices have unit norm
    # note that the order of the khatri rao product is reversed since tl.unfold has another order than assumed in paper
    for iteration in range(10 ** 4):
        A = solve_least_squares(tl.concatenate((khatri_rao([B, C]).T, V.T), axis=1).T,
                                tl.concatenate((tl.unfold(X, 0), Y), axis=1).T).T
        B = solve_least_squares(khatri_rao([A, C]), tl.unfold(X, 1).T).T
        C = solve_least_squares(khatri_rao([A, B]), tl.unfold(X, 2).T).T
        V = solve_least_squares(A, Y).T
        objective_new = 1 / 2 * tl.norm(
            X - tl.kruskal_tensor.kruskal_to_tensor((None, [A, B, C]))) ** 2 + 1 / 2 * tl.norm(
            Y - tl.kruskal_tensor.kruskal_to_tensor((None, [A, V])))
        if tl.abs(objective_new - objective_old) / objective_old <= 1e-8:
            break
        objective_old = objective_new

    tensor_3d_pred = tl.kruskal_tensor.KruskalTensor((None, [A, B, C]))
    matrix_pred = tl.kruskal_tensor.KruskalTensor((None, [A, V]))

    return tensor_3d_pred, matrix_pred
