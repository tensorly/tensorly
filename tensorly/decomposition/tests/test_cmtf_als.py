import numpy as np
import tensorly as tl

from .._cmtf_als import coupled_matrix_tensor_3d_factorization
from ...cp_tensor import cp_to_tensor
from ...random import random_cp
from ...testing import assert_


def test_coupled_matrix_tensor_3d_factorization():
    I = 21
    J = 12
    K = 16
    M = 14
    R = 3
    rng = tl.check_random_state(1234)

    tensor_cp_true = random_cp(
        (I, J, K), rank=R, normalise_factors=False, random_state=rng
    )
    matrix_cp_true = random_cp(
        (I, M), rank=R, normalise_factors=False, random_state=rng
    )
    matrix_cp_true.factors[0] = tensor_cp_true.factors[0]

    tensor_true = cp_to_tensor(tensor_cp_true)
    matrix_true = cp_to_tensor(matrix_cp_true)

    X_pred, Y_pred, errors = coupled_matrix_tensor_3d_factorization(
        tensor_true, matrix_true, R
    )

    # Check that the error monotonically decreases
    assert_(np.all(np.diff(errors) <= 0.0))

    # # Check reconstruction of noisy tensor
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor_pred = cp_to_tensor(X_pred)
    matrix_pred = cp_to_tensor(Y_pred)
    error = (
        tl.norm(tensor_true - tensor_pred) ** 2
        + tl.norm(matrix_true - matrix_pred) ** 2
    )

    assert_(error < tol_norm_2, "norm 2 of reconstruction higher than tol")

    # Test the max abs difference between the reconstruction and the tensor
    assert_(
        tl.max(tl.abs(tensor_true - tensor_pred))
        + tl.max(tl.abs(matrix_true - matrix_pred))
        < tol_max_abs,
        "abs norm of reconstruction error higher than tol",
    )


def test_coupled_matrix_tensor_3d_factorization_with_mask():
    rng = tl.check_random_state(7)
    tensor_cp = random_cp((8, 6, 5), rank=2, random_state=rng)
    matrix_cp = random_cp((8, 4), rank=2, random_state=rng)
    matrix_cp.factors[0] = tensor_cp.factors[0]
    tensor = cp_to_tensor(tensor_cp)
    matrix = cp_to_tensor(matrix_cp)
    mask = tl.tensor((rng.random_sample(tensor.shape) > 0.3).astype(float))
    observed = tensor * mask

    fitted_tensor, fitted_matrix, errors = coupled_matrix_tensor_3d_factorization(
        observed, matrix, rank=2, mask=mask, n_iter_max=200
    )
    observed_error = (
        tl.norm((tensor - cp_to_tensor(fitted_tensor)) * mask) ** 2
        + tl.norm(matrix - cp_to_tensor(fitted_matrix)) ** 2
    )
    assert_(observed_error < 1e-3)
    assert_(abs(errors[-1] - observed_error) < 1e-5)

    unmasked_tensor, unmasked_matrix, _ = coupled_matrix_tensor_3d_factorization(
        observed, matrix, rank=2, n_iter_max=200
    )
    unmasked_observed_error = (
        tl.norm((tensor - cp_to_tensor(unmasked_tensor)) * mask) ** 2
        + tl.norm(matrix - cp_to_tensor(unmasked_matrix)) ** 2
    )
    assert_(observed_error < unmasked_observed_error / 100)
