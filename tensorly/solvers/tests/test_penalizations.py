import numpy as np
import tensorly as tl

from tensorly.solvers.penalizations import (
    process_regularization_weights,
    scale_factors_fro,
)
from tensorly.testing import assert_array_equal, assert_array_almost_equal


def test_process_regularization_weights():
    # case 1: individual values for l1 l2 regularization parameters
    # nothing particular to process
    sparsity_coeffs = [1, 2, 0, 4]
    ridge_coeffs = [0, 0, 2, 0]
    n_modes = 4
    (
        out_ridge_coeffs,
        out_sp_coeffs,
        disable_rebalance,
        hom_deg,
    ) = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs,
        ridge_coefficients=ridge_coeffs,
        n_modes=n_modes,
    )
    assert_array_equal(tl.tensor(sparsity_coeffs), out_sp_coeffs)
    assert_array_equal(tl.tensor(ridge_coeffs), out_ridge_coeffs)
    assert disable_rebalance is False
    assert_array_equal(hom_deg, [1, 1, 2, 1])

    # case 2: disable works when no regularization is provided
    sparsity_coeffs = None
    ridge_coeffs = None
    n_modes = 7
    _, _, disable_rebalance, _ = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs,
        ridge_coefficients=ridge_coeffs,
        n_modes=n_modes,
    )
    assert disable_rebalance

    # case 3: add l2 regularization to a mode not regularized with a warning
    sparsity_coeffs = [1, 2, 0, 4]
    ridge_coeffs = None
    n_modes = 4
    (
        out_ridge_coeffs,
        out_sp_coeffs,
        disable_rebalance,
        hom_deg,
    ) = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs,
        ridge_coefficients=ridge_coeffs,
        n_modes=n_modes,
    )
    assert_array_equal(out_ridge_coeffs, tl.tensor([0, 0, 4, 0]))

    # case 4: rescale option is False
    sparsity_coeffs = [1, 2, 0, 4]
    ridge_coeffs = [0, 0, 3, 0]
    n_modes = 4
    (
        out_ridge_coeffs,
        out_sp_coeffs,
        disable_rebalance,
        hom_deg,
    ) = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs,
        ridge_coefficients=ridge_coeffs,
        n_modes=n_modes,
        rescale=False,
    )
    assert disable_rebalance

    # case 5: format regularizations when not provided as a list
    sparsity_coeffs = 1
    n_modes = 4
    (
        out_ridge_coeffs,
        out_sp_coeffs,
        disable_rebalance,
        hom_deg,
    ) = process_regularization_weights(
        sparsity_coefficients=sparsity_coeffs, ridge_coefficients=None, n_modes=n_modes
    )
    assert_array_equal(out_sp_coeffs, tl.tensor([1, 1, 1, 1]))
    assert disable_rebalance is False
    assert_array_equal(hom_deg, [1, 1, 1, 1])

def test_scale_factors_fro():
    # Testing only without regularizations
    factors = [tl.ones([2, 1]) for i in range(3)]
    data = -8 * tl.ones([2, 2, 2])
    cp_tensor = tl.cp_tensor.CPTensor((None, factors))
    _, scale = scale_factors_fro(cp_tensor, data, [0] * 3, [0] * 3)
    print(scale, data, factors)
    assert_array_almost_equal(scale, -2, decimal=5)
