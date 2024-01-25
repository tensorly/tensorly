import numpy as np
import tensorly as tl

from tensorly.solvers.penalizations import process_regularization_weights,cp_opt_balance, scale_factors_fro, tucker_implicit_scalar_balancing,tucker_implicit_sinkhorn_balancing
from tensorly.testing import assert_, assert_array_equal, assert_array_almost_equal
from tensorly.random import random_tucker

def test_process_regularization_weights():
    # case 1: individual values for l1 l2 regularization parameters
    # nothing particular to process
    sparsity_coeffs = [1,2,0,4]
    ridge_coeffs = [0,0,2,0]
    n_modes = 4
    out_ridge_coeffs, out_sp_coeffs, disable_rebalance, hom_deg = process_regularization_weights(
                        sparsity_coefficients=sparsity_coeffs,
                        ridge_coefficients=ridge_coeffs,
                        n_modes=n_modes)
    assert_array_equal(tl.tensor(sparsity_coeffs),out_sp_coeffs)
    assert_array_equal(tl.tensor(ridge_coeffs),out_ridge_coeffs)
    assert disable_rebalance is False
    assert_array_equal(hom_deg,[1,1,2,1])
    
    # case 2: disable works when no regularization is provided
    sparsity_coeffs = None
    ridge_coeffs = None
    n_modes = 7
    _, _, disable_rebalance, _ = process_regularization_weights(
                        sparsity_coefficients=sparsity_coeffs,
                        ridge_coefficients=ridge_coeffs,
                        n_modes=n_modes)
    assert disable_rebalance
    
    # case 3: add l2 regularization to a mode not regularized with a warning
    sparsity_coeffs = [1,2,0,4]
    ridge_coeffs = None
    n_modes = 4
    out_ridge_coeffs, out_sp_coeffs, disable_rebalance, hom_deg = process_regularization_weights(
                        sparsity_coefficients=sparsity_coeffs,
                        ridge_coefficients=ridge_coeffs,
                        n_modes=n_modes)
    assert_array_equal(out_ridge_coeffs,tl.tensor([0,0,4,0]))
    
    # case 4: rescale option is False
    sparsity_coeffs = [1,2,0,4]
    ridge_coeffs = [0,0,3,0]
    n_modes = 4
    out_ridge_coeffs, out_sp_coeffs, disable_rebalance, hom_deg = process_regularization_weights(
                        sparsity_coefficients=sparsity_coeffs,
                        ridge_coefficients=ridge_coeffs,
                        n_modes=n_modes,
                        rescale=False)
    assert disable_rebalance
    
    # case 5: format regularizations when not provided as a list
    sparsity_coeffs = 1
    n_modes = 4
    out_ridge_coeffs, out_sp_coeffs, disable_rebalance, hom_deg = process_regularization_weights(
                        sparsity_coefficients=sparsity_coeffs,
                        ridge_coefficients=None,
                        n_modes=n_modes)
    assert_array_equal(out_sp_coeffs,tl.tensor([1,1,1,1]))
    assert disable_rebalance is False
    assert_array_equal(hom_deg,[1,1,1,1])
    
def test_cp_opt_balance():
    # case 1: nonzero regs
    # at optimality after scaling, regs must satisfy
    # reg*hom_deg*(scales**hom_deg) = constant
    regs = tl.tensor([4.0,0.25,0.5,2.0])
    hom_deg = tl.tensor([1,1,2,1])
    scales = cp_opt_balance(regs=regs,hom_deg=hom_deg)
    scales = tl.tensor(scales)
    constant_v = regs*hom_deg*(scales**hom_deg)
    assert_array_almost_equal(constant_v, [1.21901365]*4, decimal=5)
    assert_array_almost_equal(tl.prod(scales), [1], decimal=5)
    
    # case 2: zero reg, optimal solution is the null scales
    regs = tl.tensor([4.0,0.25,0,2.0])
    hom_deg = tl.tensor([1,1,2,1])
    scales = cp_opt_balance(regs=regs,hom_deg=hom_deg)
    assert_array_almost_equal(tl.sum(tl.abs(tl.tensor(scales))), [0], decimal=5)
    
def test_implicit_scalar_balancing():
    core, factors = random_tucker((3, 4, 3), rank=[3, 4, 3], non_negative=True)
    # wrapper for cp_opt_balance
    # case 1: nonzero regs
    regs = tl.tensor([4.0,0.25,0.5,2.0])
    hom_deg = tl.tensor([1,1,2,1])
    _,_,scales = tucker_implicit_scalar_balancing(factors,core,regs,hom_deg)
    scales = tl.tensor(scales)
    constant_v = regs*hom_deg*(scales**hom_deg)
    assert_array_almost_equal(constant_v, [1.21901365]*4, decimal=5)
    assert_array_almost_equal(tl.prod(scales), [1], decimal=5)
    
    # case 2: zero reg, optimal solution is the null scales
    regs = tl.tensor([4.0,0.25,0,2.0])
    hom_deg = tl.tensor([1,1,2,1])
    _,_,scales = tucker_implicit_scalar_balancing(factors,core,regs,hom_deg)
    scales = tl.tensor(scales)
    assert_array_almost_equal(tl.sum(tl.abs(scales)), [0], decimal=5)
    
def test_scale_factors_fro():
    # Testing only without regularizations
    factors = [tl.ones([2,1]) for i in range(3)]
    data = -8*tl.ones([2,2,2])
    cp_tensor = tl.cp_tensor.CPTensor((None,factors))
    _, scale = scale_factors_fro(cp_tensor,data,[0]*3,[0]*3)
    print(scale, data, factors)
    assert_array_almost_equal(scale,-2,decimal=5)