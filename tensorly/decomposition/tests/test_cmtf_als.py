import numpy as np
import tensorly as tl

from .._cmtf_als import coupled_matrix_tensor_3d_factorization
from ...cp_tensor import cp_to_tensor, CPTensor
from ...testing import assert_


def test_coupled_matrix_tensor_3d_factorization():
    I = 50
    J = 50
    K = 50
    M = 50
    R = 3

    A = tl.tensor(np.random.randn(I, R))
    B = tl.tensor(np.random.randn(J, R))
    C = tl.tensor(np.random.randn(K, R))
    V = tl.tensor(np.random.randn(M, R))

    X_true = CPTensor((None, [A, B, C]))
    Y_true = CPTensor((None, [A, V]))

    X_pred, Y_pred, errors = coupled_matrix_tensor_3d_factorization(X_true, Y_true, R)

    # Check that the error monotonically decreases
    assert_(np.all(np.diff(errors) <= 1e-3))

    # Check reconstruction of noisy tensor
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor_true = cp_to_tensor(X_true)
    matrix_true = cp_to_tensor(Y_true)
    tensor_pred = cp_to_tensor(X_pred)
    matrix_pred = cp_to_tensor(Y_pred)
    error = tl.norm(tensor_true - tensor_pred) ** 2 + tl.norm(matrix_true - matrix_pred) ** 2
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(
        tl.max(tl.abs(tensor_true - tensor_pred)) + tl.max(
            tl.abs(matrix_true - matrix_pred)) < tol_max_abs,
        'abs norm of reconstruction error higher than tol')
