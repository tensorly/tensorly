import numpy as np

import tensorly as tl
from ...random import random_kruskal, check_random_state
from ..robust_decomposition import robust_pca
from ...testing import assert_array_equal, assert_, assert_array_almost_equal

def test_RPCA():
    """Test for RPCA"""
    tol = 1e-5

    sample = np.array([[1., 2, 3, 4],
                       [2, 4, 6, 8]])
    clean = np.vstack([sample[None, ...]]*100)
    noise_probability = 0.05
    rng = check_random_state(12345)
    noise = rng.choice([0., 100., -100.], size=clean.shape, replace=True,
                      p=[1 - noise_probability, noise_probability/2, noise_probability/2])
    tensor = tl.tensor(clean + noise)
    corrupted_clean = np.copy(clean)
    corrupted_noise = np.copy(noise)
    clean = tl.tensor(clean)
    noise = tl.tensor(noise)
    clean_pred, noise_pred = robust_pca(tensor, mask=None, reg_E=0.4, mu_max=10e12,
                                        learning_rate=1.2,
                                        n_iter_max=200, tol=tol, verbose=True)
    # check recovery
    assert_array_almost_equal(tensor, clean_pred+noise_pred, decimal=tol)
    # check low rank recovery
    assert_array_almost_equal(clean, clean_pred, decimal=1)
    # Check for sparsity of the gross error
    # assert tl.sum(noise_pred > 0.01) == tl.sum(noise > 0.01)
    assert_array_equal((noise_pred > 0.01), (noise > 0.01))
    # check sparse gross error recovery
    assert_array_almost_equal(noise, noise_pred, decimal=1)

    ############################
    # Test with missing values #
    ############################
    # Add some corruption (missing values, replaced by ones)
    mask = rng.choice([0, 1], clean.shape, replace=True, p=[0.05, 0.95])
    corrupted_clean[mask == 0] = 1
    tensor = tl.tensor(corrupted_clean + corrupted_noise)
    corrupted_noise = tl.tensor(corrupted_noise)
    corrupted_clean = tl.tensor(corrupted_clean)
    mask = tl.tensor(mask)
    # Decompose the tensor
    clean_pred, noise_pred = robust_pca(tensor, mask=mask, reg_E=0.4, mu_max=10e12,
                                        learning_rate=1.2,
                                        n_iter_max=200, tol=tol, verbose=True)
    # check recovery
    assert_array_almost_equal(tensor, clean_pred+noise_pred, decimal=tol)
    # check low rank recovery
    assert_array_almost_equal(corrupted_clean*mask, clean_pred*mask, decimal=1)
    # check sparse gross error recovery
    assert_array_almost_equal(noise*mask, noise_pred*mask, decimal=1)

    # Check for recovery of the corrupted/missing part
    mask = 1 - mask
    error = tl.norm((clean*mask - clean_pred*mask), 2)/tl.norm(clean*mask, 2)
    assert_(error <= 10e-3)
