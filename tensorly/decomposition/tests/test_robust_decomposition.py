from numpy.testing import assert_array_almost_equal, assert_
import numpy as np

from ...random import cp_tensor
from ...tenalg import norm
from ..robust_decomposition import robust_pca


def test_RPCA():
    """Test for RPCA"""
    # Tensor is the sum of a low rank tensor + sparse noise
    clean = cp_tensor((500, 15, 16), rank=5, full=True)
    noise = np.random.choice([0, 1, -1], size=clean.shape, replace=True, p=[0.9, 0.05, 0.05])
    tensor = clean + noise
    clean_pred, noise_pred = robust_pca(tensor, mask=None, reg_E=0.05, mu_max=10e12, n_iter_max=1000, tol=10e-12)
    # check recovery 
    assert_array_almost_equal(tensor, clean_pred+noise_pred)
    # check low rank recovery
    assert_array_almost_equal(clean, clean_pred)
    # check sparse gross error recovery
    assert_array_almost_equal(noise, noise_pred)

    ## Test with missing values
    clean = cp_tensor((500, 15, 16), rank=5, full=True)
    noise = np.random.choice([0, 1, -1], size=clean.shape, replace=True, p=[0.9, 0.05, 0.05])
    corrupted_clean = np.copy(clean)
    # Add some corruption (missing values, replaced by ones)
    mask = np.random.choice([False, True], clean.shape, replace=True, p=[0.05, 0.95])
    corrupted_clean[~mask] = 1
    tensor = corrupted_clean + noise
    # Decompose the tensor
    clean_pred, noise_pred = robust_pca(tensor, mask=mask, reg_E=0.05, mu_max=10e12, n_iter_max=1000, tol=10e-12)
    # check recovery 
    assert_array_almost_equal(tensor, clean_pred+noise_pred)
    # check low rank recovery
    assert_array_almost_equal(corrupted_clean[mask], clean_pred[mask])
    # check sparse gross error recovery
    assert_array_almost_equal(noise[mask], noise_pred[mask])

    tol = 10e-12
    # Check for recovery of the corrupted/missing part
    error = norm((clean[~mask] - clean_pred[~mask]), 2)/norm(clean[~mask], 2)
    assert_(error <= tol)

