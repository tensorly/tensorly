import numpy as np

import tensorly as tl
from ..cmtf_als import (factor_match_score_3d, coupled_matrix_tensor_3d_factorization)
from ...kruskal_tensor import KruskalTensor
from ...testing import assert_array_almost_equal, assert_array_less


def test_factor_match_score_3d():
    # same factors
    X_true = KruskalTensor((None, [tl.tensor([[4, 1], [0, 2]]), tl.tensor([[0, 1], [3, 5]]),
                                   tl.tensor([[1, 1], [1, 2]])]))
    Y_true = KruskalTensor((None, [tl.tensor([[4, 1], [0, 2]]), tl.tensor([[1, 1], [1, 2]])]))
    assert_array_almost_equal(1, factor_match_score_3d(X_true, Y_true, X_true, Y_true))

    # with permutation
    X_pred = KruskalTensor((None, [tl.tensor([[1, 4], [2, 0]]), tl.tensor([[1, 0], [5, 3]]),
                                   tl.tensor([[1, 1], [2, 1]])]))
    Y_pred = KruskalTensor((None, [tl.tensor([[1, 4], [2, 0]]), tl.tensor([[1, 1], [2, 1]])]))
    assert_array_almost_equal(1, factor_match_score_3d(X_true, Y_true, X_pred, Y_pred))


def test_coupled_matrix_tensor_3d_factorization():
    # simple scenario (without noise)
    A = tl.tensor([[1, 2], [3, 4]])
    B = tl.tensor([[1, 0], [0, 2]])
    C = tl.tensor([[2, 0], [0, 1]])
    V = tl.tensor([[2, 0], [0, 1]])
    R = 2
    X_true = KruskalTensor((None, [A, B, C]))
    Y_true = KruskalTensor((None, [A, V]))
    X_pred, Y_pred = coupled_matrix_tensor_3d_factorization(X_true, Y_true, R)
    assert_array_less(0.99 ** 4, factor_match_score_3d(X_true, Y_true, X_pred, Y_pred))

    # scenario as in paper (with noise)

    I = 50
    J = 50
    K = 50
    M = 50
    R = 3
    eta = 0.1

    A = tl.tensor(np.random.randn(I, R))
    B = tl.tensor(np.random.randn(J, R))
    C = tl.tensor(np.random.randn(K, R))
    V = tl.tensor(np.random.randn(M, R))
    N_tens = tl.tensor(np.random.randn(I, J, K))
    N = tl.tensor(np.random.randn(I, M))

    X_true = KruskalTensor((None, [A, B, C]))
    X = tl.kruskal_tensor.kruskal_to_tensor(X_true)
    X = X + eta * N_tens * tl.norm(X) / tl.norm(N_tens)
    Y_true = KruskalTensor((None, [A, V]))
    Y = tl.kruskal_tensor.kruskal_to_tensor(Y_true)
    Y = Y + eta * N * tl.norm(Y) / tl.norm(N)

    X_pred, Y_pred = coupled_matrix_tensor_3d_factorization(X, Y, R)
    assert_array_less(0.99 ** 4, factor_match_score_3d(X_true, Y_true, X_pred, Y_pred))
