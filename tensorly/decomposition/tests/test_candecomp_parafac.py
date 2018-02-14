import numpy as np
import pytest

from ..candecomp_parafac import (
    parafac, non_negative_parafac, normalize_factors, initialize_factors)
from ...kruskal_tensor import kruskal_to_tensor
from ...random import check_random_state
from ... import backend as T


randomized_test_cases = {
    'order3_small': {
        'dimensions': (8, 8, 8), 'rank': 4, 'seed': 0},
    'order3_large': {
        'dimensions': (16, 16, 16), 'rank': 12, 'seed': 0},
    'order3_mixed': {
        'dimensions': (16, 12, 8), 'rank': 8, 'seed': 0},
    'order3_small_orthogonal': {
        'dimensions': (8, 8, 8), 'rank': 4, 'seed': 0, 'orthogonal': True},
    'order3_large_orthogonal': {
        'dimensions': (16, 16, 16), 'rank': 12, 'seed': 0, 'orthogonal': True},
    'order3_mixed_orthogonal': {
        'dimensions': (16, 12, 8), 'rank': 8, 'seed': 0, 'orthogonal': True},
    'order4': {
        'dimensions': (8, 8, 8, 8), 'rank': 4, 'seed': 0},
    'order4_orthogonal': {
        'dimensions': (8, 8, 8, 8), 'rank': 4, 'seed': 0, 'orthogonal': True},
}


def create_random_decomposition(dimensions=(10,10,10), rank=10, seed=None,
                                orthogonal=False, noisy=False):
    r"""Random tensor generation used in testing.

    Returns a random tensor of specified order along with a list of its
    constituent normalized factor matrices and corresponding weight vector.

    Parameters
    ----------
    dimensions : tuple
    rank : int
        The desired rank of the tensor. Note that if `orthogonal == True` then
        the rank must be less than or equal to the smallest dimension of the
        tensor.
    seed : int
        Random number generation seed.
    orthogonal : bool
        If `True`, creates a tensor with orthogonal rank-1 components.
    noisy : bool
        If `True`, adds Gaussian noise to the tensor. (Useful when constructing
        near-orthogonal tensors for experimentation.)

    Returns
    -------
    tensor : ndarray
    factors : list of 2D ndarrays
        A list of random ALS factors.
    weights : 1D ndarray
        A list of random weights.

    """
    order = len(dimensions)
    dim = min(dimensions)
    if (rank > dim) and orthogonal:
        raise ValueError('Can only construct orthogonal tensors when '
                         'rank <= min(dimensions)')

    np.random.seed(seed)
    weights = 4*T.arange(1, rank+1)
    factors = [T.tensor(np.random.randn(dim,rank)) for dim in dimensions]

    if orthogonal:
        factors = [T.qr(factor)[0] for factor in factors]
    tensor = kruskal_to_tensor(factors, weights)

    if noisy:
        scale = rank/(rank+10.**(order-1))
        tensor += T.tensor(scale*np.random.randn(*dimensions))
    return tensor, factors, weights


def test_parafac_errors():
    with pytest.raises(ValueError) as e:
        create_random_decomposition(dimensions=(8,8,8), rank=10, orthogonal=True)

    tensor = T.tensor([[1,2],[3,4]])
    rank = 1
    with pytest.raises(ValueError) as e:
        initialize_factors(tensor, rank, init='invalid initialization')


@pytest.mark.parametrize(
    'params',
    list(randomized_test_cases.values()),
    ids=list(randomized_test_cases.keys()))
def test_parafac_random(params):
    rank = params['rank']
    tensor, weights, factors = create_random_decomposition(**params)
    factors_estimated = parafac(tensor, rank, n_iter_max=500, tol=1e-8)
    factors_estimated, weights_estimated = normalize_factors(factors_estimated)
    tensor_estimted = kruskal_to_tensor(
        factors_estimated, weights=weights_estimated)

    error = T.norm(tensor - tensor_estimted)/T.norm(tensor)
    T.assert_(error < 1e-4,
              '2-Norm relative error between known and recovered tensor too large')

@pytest.mark.parametrize(
    'params',
    list(randomized_test_cases.values()),
    ids=list(randomized_test_cases.keys()))
def test_parafac_random_noisy(params):
    rank = params['rank']
    tensor, weights, factors = create_random_decomposition(noisy=True, **params)
    factors_estimated = parafac(tensor, rank, n_iter_max=500, tol=1e-8)
    factors_estimated, weights_estimated = normalize_factors(factors_estimated)
    tensor_estimted = kruskal_to_tensor(
        factors_estimated, weights=weights_estimated)

    error = T.norm(tensor - tensor_estimted)/T.norm(tensor)
    T.assert_(error < 1e-1,
              '2-Norm relative error between known and recovered tensor too large')


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
    T.assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    T.assert_(T.max(T.abs(rec_svd - tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    T.assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')


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
        T.assert_(T.all(factor >= 0))

    reconstructed_tensor = kruskal_to_tensor(factors)
    nn_reconstructed_tensor = kruskal_to_tensor(nn_factors)
    error = T.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= T.norm(reconstructed_tensor, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    T.assert_(T.max(T.abs(reconstructed_tensor - nn_reconstructed_tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    factors_svd = non_negative_parafac(tensor, rank=3, n_iter_max=100, tol=10e-4,
                                       init='svd')
    factors_random = non_negative_parafac(tensor, rank=3, n_iter_max=100, tol=10e-4,
                                          init='random', random_state=1234, verbose=0)
    rec_svd = kruskal_to_tensor(factors_svd)
    rec_random = kruskal_to_tensor(factors_random)
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    T.assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')

