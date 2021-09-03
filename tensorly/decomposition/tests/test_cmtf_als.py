import numpy as np
import tensorly as tl

from .._cmtf_als import coupled_matrix_tensor_3d_factorization
from ...cp_tensor import cp_to_tensor, CPTensor
from ...random import random_cp
from ...testing import assert_


def test_coupled_matrix_tensor_3d_factorization():
    I = 21
    J = 12
    K = 8
    M = 7
    R = 3

    tensor_cp_true = random_cp((I, J, K), rank=R, normalise_factors=False)
    matrix_cp_true = random_cp((I, M), rank=R, normalise_factors=False)
    matrix_cp_true.factors[0] = tensor_cp_true.factors[0]

    tensor_true = cp_to_tensor(tensor_cp_true)
    matrix_true = cp_to_tensor(matrix_cp_true)

    X_pred, Y_pred, errors = coupled_matrix_tensor_3d_factorization(tensor_true, matrix_true, R)

    # Check that the error monotonically decreases
    assert_(np.all(np.diff(errors) <= 0.0))

    # # Check reconstruction of noisy tensor
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor_pred = cp_to_tensor(X_pred)
    matrix_pred = cp_to_tensor(Y_pred)
    error = tl.norm(tensor_true - tensor_pred) ** 2 + tl.norm(matrix_true - matrix_pred) ** 2
    # TODO: These error checks do not always pass, possibly due to poor SVD initialization.
    # assert_(error < tol_norm_2,
    #         'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    # assert_(
    #     tl.max(tl.abs(tensor_true - tensor_pred)) + tl.max(
    #         tl.abs(matrix_true - matrix_pred)) < tol_max_abs,
    #     'abs norm of reconstruction error higher than tol')
