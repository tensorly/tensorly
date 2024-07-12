import numpy as np
import tensorly as tl

from tensorly.solvers.penalizations import process_regularization_weights
from tensorly.testing import assert_array_equal


def test_process_regularization_weights():
    """
    Test of process_regularization_weights.
    Process_regularization_weights take two lists as inputs and checks that
        - no mode is unregularized (the constrained decomposition problem would be ill-posed)
        - turns float inputs into lists
    This test checks that the output of the method matches a manual processing.
    """
    # case 1: individual values for l1 l2 regularization parameters
    # nothing particular to process
    sparsity_coeffs = [1, 2, 0, 4]
    ridge_coeffs = [0, 0, 2, 0]
    n_modes = 4
    (
        out_ridge_coeffs,
        out_sp_coeffs,
    ) = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs,
        ridge_coefficients=ridge_coeffs,
        n_modes=n_modes,
    )
    assert_array_equal(tl.tensor(sparsity_coeffs), out_sp_coeffs)
    assert_array_equal(tl.tensor(ridge_coeffs), out_ridge_coeffs)

    # case 2: add l2 regularization to a mode not regularized with a warning
    sparsity_coeffs = [1, 2, 0, 4]
    ridge_coeffs = None
    n_modes = 4
    (
        out_ridge_coeffs,
        out_sp_coeffs,
    ) = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs,
        ridge_coefficients=ridge_coeffs,
        n_modes=n_modes,
    )
    assert_array_equal(out_ridge_coeffs, tl.tensor([0, 0, 4, 0]))

    # case 3: format regularizations when not provided as a list
    sparsity_coeffs = 1
    n_modes = 4
    (
        out_ridge_coeffs,
        out_sp_coeffs,
    ) = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs, ridge_coefficients=None, n_modes=n_modes
    )
    assert_array_equal(out_sp_coeffs, tl.tensor([1, 1, 1, 1]))
