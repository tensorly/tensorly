from scipy.sparse.linalg import svds
from scipy.linalg import svd
from numpy.testing import assert_array_almost_equal, assert_raises
import numpy as np
from .._partial_svd import partial_svd


def test_partial_svd():
    """Test for partial_svd"""
    sizes = [(100, 100), (100, 5), (10, 10), (5, 100)]
    n_eigenvecs = [10, 4, 5, 4]

    # Compare with sparse SVD
    for s, n in zip(sizes, n_eigenvecs):
        matrix = np.random.random(s)
        fU, fS, fV = partial_svd(matrix, n_eigenvecs=n)
        U, S, V = svds(matrix, k=n, which='LM')
        U, S, V = U[:, ::-1], S[::-1], V[::-1, :]
        assert_array_almost_equal(np.abs(S), np.abs(fS))
        assert_array_almost_equal(np.abs(U), np.abs(fU))
        assert_array_almost_equal(np.abs(V), np.abs(fV))

    # Compare with standard SVD
    sizes = [(100, 100), (100, 5), (10, 10), (10, 4), (5, 100)]
    n_eigenvecs = [10, 4, 5, 4, 4]
    for s, n in zip(sizes, n_eigenvecs):
        matrix = np.random.random(s)
        fU, fS, fV = partial_svd(matrix, n_eigenvecs=n)

        U, S, V = svd(matrix)
        U, S, V = U[:, :n], S[:n], V[:n, :]
        # Test for SVD
        assert_array_almost_equal(np.abs(S), np.abs(fS))
        assert_array_almost_equal(np.abs(U), np.abs(fU))
        assert_array_almost_equal(np.abs(V), np.abs(fV))

    with assert_raises(ValueError):
        tensor = np.random.random((3, 3, 3))
        partial_svd(tensor)
