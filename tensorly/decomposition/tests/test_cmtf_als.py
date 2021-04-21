import numpy as np
import tensorly as tl

from ..cmtf_als import coupled_matrix_tensor_3d_factorization
from ...cp_tensor import cp_to_tensor, CPTensor
from ...testing import assert_


def test_coupled_matrix_tensor_3d_factorization():
    # # simple scenario (without noise)
    # A = tl.tensor([[1, 2], [3, 4]])
    # B = tl.tensor([[1, 0], [0, 2]])
    # C = tl.tensor([[2, 0], [0, 1]])
    # V = tl.tensor([[2, 0], [0, 1]])
    # R = 2
    # X_true = CPTensor((None, [A, B, C]))
    # Y_true = CPTensor((None, [A, V]))
    # X_pred, Y_pred, errors = coupled_matrix_tensor_3d_factorization(X_true, Y_true, R)
    #
    # # Check that the error monotonically decreases
    # assert_(np.all(np.diff(errors) <= 0.0))
    #
    # # Check reconstruction
    # tol_norm_2 = 10e-2
    # tol_max_abs = 10e-2
    # tensor_pred = cp_to_tensor(X_pred)
    # tensor = cp_to_tensor(X_true)
    # error = tl.norm(tensor_pred - X_true, 2)
    # error /= tl.norm(tensor, 2)
    # assert_(error < tol_norm_2,
    #         'norm 2 of reconstruction higher than tol')
    # # Test the max abs difference between the reconstruction and the tensor
    # assert_(tl.max(tl.abs(tensor_pred - tensor)) < tol_max_abs,
    #         'abs norm of reconstruction error higher than tol')
    #
    # # fms = factor_match_score_3d(X_true, Y_true, X_pred, Y_pred)
    # # assert_array_less(0.99 ** 4, fms)

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

    X_true = CPTensor((None, [A, B, C]))
    X = tl.cp_tensor.cp_to_tensor(X_true)
    X = X + eta * N_tens * tl.norm(X) / tl.norm(N_tens)
    Y_true = CPTensor((None, [A, V]))
    Y = tl.cp_tensor.cp_to_tensor(Y_true)
    Y = Y + eta * N * tl.norm(Y) / tl.norm(N)

    X_pred, Y_pred, errors = coupled_matrix_tensor_3d_factorization(X, Y, R)

    # Check that the error monotonically decreases
    assert_(np.all(np.diff(errors) <= 0.0))

    # Check reconstruction
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor_pred = cp_to_tensor(X_pred)
    tensor = cp_to_tensor(X_true)
    error = tl.norm(tensor_pred - X_true, 2)
    error /= tl.norm(tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(tl.max(tl.abs(tensor_pred - tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    # fms = factor_match_score_3d(X_true, Y_true, X_pred, Y_pred)
    # assert_array_less(0.99 ** 4, fms)
