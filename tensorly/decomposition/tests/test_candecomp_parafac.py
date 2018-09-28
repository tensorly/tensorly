import numpy as np
import pytest

import tensorly as tl
from tensorly.decomposition.candecomp_parafac import (
    parafac, non_negative_parafac, initialize_factors,
    sample_khatri_rao, randomised_parafac)
from tensorly.kruskal_tensor import kruskal_to_tensor
from tensorly.random import check_random_state, cp_tensor
from tensorly.tenalg import khatri_rao
from tensorly import backend as T
from tensorly.testing import assert_array_equal, assert_


def test_parafac():
    """Test for the CANDECOMP-PARAFAC decomposition
    """
    rng = check_random_state(1234)
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor = T.tensor(rng.random_sample((3, 4, 2)))
    factors_svd = parafac(tensor, rank=4, n_iter_max=200, init='svd', tol=10e-5)
    factors_random = parafac(tensor, rank=4, n_iter_max=200, init='random', tol=10e-5, random_state=1234, verbose=0)
    rec_svd = kruskal_to_tensor(factors_svd)
    rec_random = kruskal_to_tensor(factors_random)
    error = T.norm(rec_svd - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.max(T.abs(rec_svd - tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')

    with np.testing.assert_raises(ValueError):
        rank = 4
        _ = initialize_factors(tensor, rank, init='bogus init type')


def test_non_negative_parafac():
    """Test for non-negative PARAFAC

    TODO: more rigorous test
    """
    tol_norm_2 = 10e-1
    tol_max_abs = 1
    rng = check_random_state(1234)
    tensor = T.tensor(rng.random_sample((3, 3, 3))+1)
    factors = parafac(tensor, rank=3, n_iter_max=120)
    nn_factors = non_negative_parafac(tensor, rank=3, n_iter_max=100, tol=10e-4, init='svd', verbose=0)

    # Make sure all components are positive
    for factor in nn_factors:
        assert_(T.all(factor >= 0))

    reconstructed_tensor = kruskal_to_tensor(factors)
    nn_reconstructed_tensor = kruskal_to_tensor(nn_factors)
    error = T.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= T.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.max(T.abs(reconstructed_tensor - nn_reconstructed_tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    factors_svd = non_negative_parafac(tensor, rank=3, n_iter_max=100,
                                       tol=10e-4, init='svd')
    factors_random = non_negative_parafac(tensor, rank=3, n_iter_max=100, tol=10e-4,
                                          init='random', random_state=1234, verbose=0)
    rec_svd = kruskal_to_tensor(factors_svd)
    rec_random = kruskal_to_tensor(factors_random)
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')


@pytest.mark.xfail(tl.get_backend() == 'tensorflow', reason='Fails on tensorflow')
def test_sample_khatri_rao():
    """ Test for sample_khatri_rao
    """

    rng = check_random_state(1234)
    t_shape = (8, 9, 10)
    rank = 3
    tensor = T.tensor(rng.random_sample(t_shape)+1)
    factors = parafac(tensor, rank=rank, n_iter_max=120)
    num_samples = 4
    skip_matrix = 1
    sampled_kr, sampled_indices, sampled_rows = sample_khatri_rao(factors, num_samples, skip_matrix=skip_matrix,
                                                                  return_sampled_rows=True)
    assert_(T.shape(sampled_kr) == (num_samples, rank),
              'Sampled shape of khatri-rao product is inconsistent')
    assert_(np.max(sampled_rows) < (t_shape[0] * t_shape[2]),
              'Largest sampled row index is bigger than number of columns of'
              'unfolded matrix')
    assert_(np.min(sampled_rows) >= 0,
              'Smallest sampled row index index is smaller than 0')
    true_kr = khatri_rao(factors, skip_matrix=skip_matrix)
    for ix, j in enumerate(sampled_rows):
        assert_array_equal(true_kr[j], sampled_kr[int(ix)], err_msg='Sampled khatri_rao product doesnt correspond to product')


@pytest.mark.xfail(tl.get_backend() == 'tensorflow', reason='Fails on tensorflow')
def test_randomised_parafac():
    """ Test for randomised_parafac
    """
    rng = check_random_state(1234)
    t_shape = (10, 10, 10)
    n_samples = 8
    tensor = T.tensor(rng.random_sample(t_shape))
    rank = 4
    factors_svd = randomised_parafac(tensor, rank, n_samples, n_iter_max=1000,
                                     init='svd', tol=10e-5, verbose=True)
    for i, f in enumerate(factors_svd):
        assert_(T.shape(f) == (t_shape[i], rank),
                  'Factors are of incorrect size')

    # test tensor reconstructed properly
    tolerance = 0.05
    tensor = cp_tensor(shape=(10, 10, 10), rank=4, full=True)
    factors = randomised_parafac(tensor, rank=5, n_samples=100, max_stagnation=20, n_iter_max=100, tol=0, verbose=0)
    reconstruction = kruskal_to_tensor(factors)
    error = float(T.norm(reconstruction - tensor, 2)/T.norm(tensor, 2))
    assert_(error < tolerance, msg='reconstruction of {} (higher than tolerance of {})'.format(error, tolerance))
