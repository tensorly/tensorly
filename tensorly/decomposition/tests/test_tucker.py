from numpy.testing import assert_, assert_equal, assert_array_almost_equal
import numpy as np
from .._tucker import tucker, non_negative_tucker
from ...tucker import tucker_to_tensor
from ...tenalg import norm
from ...utils import check_random_state


def test_tucker():
    """Test for the Tucker decomposition"""
    rng = check_random_state(1234)

    tol_norm_2 = 10e-3
    tol_max_abs = 10e-1
    tensor = rng.random_sample((3, 4, 3))
    core, factors = tucker(tensor, ranks=None, n_iter_max=200, verbose=True)
    reconstructed_tensor = tucker_to_tensor(core, factors)
    norm_rec = norm(reconstructed_tensor, 2)
    norm_tensor = norm(tensor, 2)
    assert_((norm_rec - norm_tensor)/norm_rec < tol_norm_2)

    # Test the max abs difference between the reconstruction and the tensor
    assert_(np.max(np.abs(norm_rec - norm_tensor)) < tol_max_abs)

    # Test the shape of the core and factors
    ranks = [2, 3, 1]
    core, factors = tucker(tensor, ranks=ranks, n_iter_max=100, verbose=1)
    for i, rank in enumerate(ranks):
        assert_equal(factors[i].shape, (tensor.shape[i], ranks[i]),
                     err_msg="factors[{}].shape={}, expected {}".format(
                         i, factors[i].shape, (tensor.shape[i], ranks[i])))
        assert_equal(core.shape[i], rank, err_msg="Core.shape[{}]={}, "
                     "expected {}".format(i, core.shape[i], rank))

    # Random and SVD init should converge to a similar solution
    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1

    core_svd, factors_svd = tucker(tensor, ranks=[3, 4, 3], n_iter_max=200, init='svd', verbose=1)
    core_random, factors_random = tucker(tensor, ranks=[3, 4, 3], n_iter_max=200, init='random')
    rec_svd = tucker_to_tensor(core_svd, factors_svd)
    rec_random = tucker_to_tensor(core_random, factors_random)
    error = norm(rec_svd - rec_random, 2)
    error /= norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(np.max(np.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')


def test_non_negative_tucker():
    """Test for non-negative Tucker"""
    rng = check_random_state(1234)

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    tensor = rng.random_sample((3, 4, 3)) + 1
    core, factors = tucker(tensor, ranks=[3, 4, 3], n_iter_max=200, verbose=1)
    nn_core, nn_factors = non_negative_tucker(tensor, ranks=[3, 4, 3], n_iter_max=100)

    # Make sure all components are positive
    for factor in nn_factors:
        assert_(np.all(factor >= 0))
    assert_(np.all(nn_core >= 0))

    reconstructed_tensor = tucker_to_tensor(core, factors)
    nn_reconstructed_tensor = tucker_to_tensor(nn_core, nn_factors)
    error = norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction error higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    assert_(np.max(np.abs(reconstructed_tensor - nn_reconstructed_tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    core_svd, factors_svd = non_negative_tucker(tensor, ranks=[3, 4, 3], n_iter_max=500, init='svd', verbose=1)
    core_random, factors_random = non_negative_tucker(tensor, ranks=[3, 4, 3], n_iter_max=200, init='random')
    rec_svd = tucker_to_tensor(core_svd, factors_svd)
    rec_random = tucker_to_tensor(core_random, factors_random)
    error = norm(rec_svd - rec_random, 2)
    error /= norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(np.max(np.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')