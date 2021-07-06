import numpy as np
import tensorly as tl
from .._tucker import tucker, partial_tucker, non_negative_tucker, non_negative_tucker_hals, Tucker, Tucker_NN
from ...tucker_tensor import tucker_to_tensor
from ...tenalg import multi_mode_dot
from ...random import random_tucker
from ...testing import assert_equal, assert_, assert_array_equal, assert_class_wrapper_correctly_passes_arguments


def test_partial_tucker():
    """Test for the Partial Tucker decomposition"""
    rng = tl.check_random_state(1234)
    tol_norm_2 = 10e-3
    tol_max_abs = 10e-1
    tensor = tl.tensor(rng.random_sample((3, 4, 3)))
    modes = [1, 2]
    core, factors = partial_tucker(tensor, modes, rank=None, n_iter_max=200, verbose=True)
    reconstructed_tensor = multi_mode_dot(core, factors, modes=modes)
    norm_rec = tl.norm(reconstructed_tensor, 2)
    norm_tensor = tl.norm(tensor, 2)
    assert_((norm_rec - norm_tensor)/norm_rec < tol_norm_2)

    # Test the max abs difference between the reconstruction and the tensor
    assert_(tl.max(tl.abs(norm_rec - norm_tensor)) < tol_max_abs)

    # Test the shape of the core and factors
    ranks = [3, 1]
    core, factors = partial_tucker(tensor, modes=modes, rank=ranks, n_iter_max=100, verbose=1)
    for i, rank in enumerate(ranks):
        assert_equal(factors[i].shape, (tensor.shape[i+1], ranks[i]),
                     err_msg="factors[{}].shape={}, expected {}".format(
                         i, factors[i].shape, (tensor.shape[i+1], ranks[i])))
    assert_equal(core.shape, [tensor.shape[0]]+ranks, err_msg="Core.shape={}, "
                     "expected {}".format(core.shape, [tensor.shape[0]]+ranks))

    # Test random_state fixes the core and the factor matrices
    core1, factors1 = partial_tucker(tensor, modes=modes, rank=ranks, random_state=0)
    core2, factors2 = partial_tucker(tensor, modes=modes, rank=ranks, random_state=0)
    assert_array_equal(core1, core2)
    for factor1, factor2 in zip(factors1, factors2):
        assert_array_equal(factor1, factor2)


def test_tucker(monkeypatch):
    """Test for the Tucker decomposition"""
    rng = tl.check_random_state(1234)

    tol_norm_2 = 10e-3
    tol_max_abs = 10e-1
    tensor = tl.tensor(rng.random_sample((3, 4, 3)))
    core, factors = tucker(tensor, rank=None, n_iter_max=200, verbose=True)
    reconstructed_tensor = tucker_to_tensor((core, factors))
    norm_rec = tl.norm(reconstructed_tensor, 2)
    norm_tensor = tl.norm(tensor, 2)
    assert((norm_rec - norm_tensor)/norm_rec < tol_norm_2)

    # Test the max abs difference between the reconstruction and the tensor
    assert(tl.max(tl.abs(reconstructed_tensor - tensor)) < tol_max_abs)

    # Test the shape of the core and factors
    ranks = [2, 3, 1]
    core, factors = tucker(tensor, rank=ranks, n_iter_max=100, verbose=1)
    for i, rank in enumerate(ranks):
        assert_equal(factors[i].shape, (tensor.shape[i], ranks[i]),
                     err_msg="factors[{}].shape={}, expected {}".format(
                         i, factors[i].shape, (tensor.shape[i], ranks[i])))
        assert_equal(tl.shape(core)[i], rank, err_msg="Core.shape[{}]={}, "
                     "expected {}".format(i, core.shape[i], rank))

    # try fixing the core
    factors_init = [tl.copy(f) for f in factors]
    _, factors = tucker(tensor, rank=ranks, init=(core, factors), fixed_factors=[1], n_iter_max=100, verbose=1)
    assert_array_equal(factors[1], factors_init[1])

    # Random and SVD init should converge to a similar solution
    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1

    core_svd, factors_svd = tucker(tensor, rank=[3, 4, 3], n_iter_max=200, init='svd', verbose=1)
    core_random, factors_random = tucker(tensor, rank=[3, 4, 3], n_iter_max=200, init='random', random_state=1234)
    rec_svd = tucker_to_tensor((core_svd, factors_svd))
    rec_random = tucker_to_tensor((core_random, factors_random))
    error = tl.norm(rec_svd - rec_random, 2)
    error /= tl.norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(tl.max(tl.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, tucker, Tucker, ignore_args={}, rank=3)

def test_masked_tucker():
    """Test for the masked Tucker decomposition.
    This checks that a mask of 1's is identical to the unmasked case.
    """
    rng = tl.check_random_state(1234)
    tensor = tl.tensor(rng.random_sample((3, 3, 3)))
    mask = tl.tensor(np.ones((3, 3, 3)))

    mask_fact = tucker(tensor, rank=(2, 2, 2), mask=mask)
    fact = tucker(tensor, rank=(2, 2, 2))
    diff = tucker_to_tensor(mask_fact) - tucker_to_tensor(fact)
    assert_(tl.norm(diff) < 0.001, 'norm 2 of reconstruction higher than 0.001')

    # Mask an outlier value, and check that the decomposition ignores it
    tensor = random_tucker((5, 5, 5), (1, 1, 1), full=True, random_state=1234)
    mask = tl.tensor(np.ones((5, 5, 5)))

    mask_tensor = tl.tensor(tensor)
    mask_tensor = tl.index_update(mask_tensor, tl.index[0, 0, 0], 1.0)
    mask = tl.index_update(mask, tl.index[0, 0, 0], 0)

    # We won't use the SVD decomposition, but check that it at least runs successfully
    mask_fact = tucker(mask_tensor, rank=(1, 1, 1), mask=mask, init="svd")
    mask_fact = tucker(mask_tensor, rank=(1, 1, 1), mask=mask, init="random", random_state=1234)
    mask_err = tl.norm(tucker_to_tensor(mask_fact) - tensor)

    assert_(mask_err < 0.001, 'norm 2 of reconstruction higher than 0.001')

def test_non_negative_tucker(monkeypatch):
    """Test for non-negative Tucker"""
    rng = tl.check_random_state(1234)

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    tensor = tl.tensor(rng.random_sample((3, 4, 3)) + 1)
    core, factors = tucker(tensor, rank=[3, 4, 3], n_iter_max=200, verbose=1)
    nn_core, nn_factors = non_negative_tucker(tensor, rank=[3, 4, 3], n_iter_max=100)

    # Make sure all components are positive
    for factor in nn_factors:
        assert_(tl.all(factor >= 0))
    assert_(tl.all(nn_core >= 0))

    reconstructed_tensor = tucker_to_tensor((core, factors))
    nn_reconstructed_tensor = tucker_to_tensor((nn_core, nn_factors))
    error = tl.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= tl.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction error higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    assert_(tl.norm(reconstructed_tensor - nn_reconstructed_tensor, 'inf') < tol_max_abs,
              'abs norm of reconstruction error higher than tol')

    core_svd, factors_svd = non_negative_tucker(tensor, rank=[3, 4, 3], n_iter_max=500, init='svd', verbose=1)
    core_random, factors_random = non_negative_tucker(tensor, rank=[3, 4, 3], n_iter_max=200, init='random', random_state=1234)
    rec_svd = tucker_to_tensor((core_svd, factors_svd))
    rec_random = tucker_to_tensor((core_random, factors_random))
    error = tl.norm(rec_svd - rec_random, 2)
    error /= tl.norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(tl.norm(rec_svd - rec_random, 'inf') < tol_max_abs,
            'abs norm of difference between svd and random init too high')

    # Test for a single rank passed
    # (should be used for all modes)
    rank = 3
    target_shape = (rank, )*tl.ndim(tensor)
    core, factors = non_negative_tucker(tensor, rank=rank)
    assert_(tl.shape(core) == target_shape, 'core has the wrong shape, got {}, but expected {}.'.format(tl.shape(core), target_shape))
    for i, f in enumerate(factors):
        expected_shape = (tl.shape(tensor)[i], rank)
        assert_(tl.shape(f) == expected_shape, '{}-th factor has the wrong shape, got {}, but expected {}.'.format(
                i, tl.shape(f), expected_shape))

    assert_class_wrapper_correctly_passes_arguments(monkeypatch, non_negative_tucker, Tucker_NN, ignore_args={'return_errors'}, rank=3)


def test_non_negative_tucker_hals():
    """Test for non-negative Tucker wih HALS"""
    rng = tl.check_random_state(1234)

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    tensor = tl.tensor(rng.random_sample((3, 4, 3)) + 1)
    core, factors = tucker(tensor, rank=[3, 4, 3], n_iter_max=200)
    nn_core, nn_factors = non_negative_tucker_hals(tensor, rank=[3, 4, 3], n_iter_max=100)

    # Make sure all components are positive
    for factor in nn_factors:
        assert_(tl.all(factor >= 0))
    assert_(tl.all(nn_core >= 0))

    reconstructed_tensor = tucker_to_tensor((core, factors))
    nn_reconstructed_tensor = tucker_to_tensor((nn_core, nn_factors))
    error = tl.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= tl.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction error higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    assert_(tl.norm(reconstructed_tensor - nn_reconstructed_tensor, 'inf') < tol_max_abs,
              'abs norm of reconstruction error higher than tol')

    core_svd, factors_svd = non_negative_tucker_hals(tensor, rank=[3, 4, 3], n_iter_max=500, init='svd')
    core_random, factors_random = non_negative_tucker_hals(tensor, rank=[3, 4, 3], n_iter_max=200, init='random',           random_state=1234)
    rec_svd = tucker_to_tensor((core_svd, factors_svd))
    rec_random = tucker_to_tensor((core_random, factors_random))
    error = tl.norm(rec_svd - rec_random, 2)
    error /= tl.norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(tl.norm(rec_svd - rec_random, 'inf') < tol_max_abs,
            'abs norm of difference between svd and random init too high')

    # Test for a single rank passed
    # (should be used for all modes)
    rank = 3
    target_shape = (rank, )*tl.ndim(tensor)
    core, factors = non_negative_tucker_hals(tensor, rank=rank)
    assert_(tl.shape(core) == target_shape, 'core has the wrong shape, got {}, but expected {}.'.format(tl.shape(core), target_shape))
    for i, f in enumerate(factors):
        expected_shape = (tl.shape(tensor)[i], rank)
        assert_(tl.shape(f) == expected_shape, '{}-th factor has the wrong shape, got {}, but expected {}.'.format(
                i, tl.shape(f), expected_shape))