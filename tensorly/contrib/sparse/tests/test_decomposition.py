from ..decomposition import parafac, robust_pca
from ..tenalg import multi_mode_dot
from ..cp_tensor import cp_to_tensor
from .... import backend as tl
from ..backend import sparse_context

import pytest

if not tl.get_backend() == "numpy":
    pytest.skip("Tests for sparse only with numpy backend", allow_module_level=True)
pytest.importorskip("sparse")

import sparse
import numpy as np


def test_sparse_parafac():
    """Test for sparse parafac"""
    # Make sure the algorithm stays sparse. This will run out of memory on
    # most machines if the algorithm densifies.
    random_state = 1234
    rank = 3
    factors = [
        sparse.random((300, rank), random_state=random_state),
        sparse.random((1500, rank), random_state=random_state),
    ]
    weights = np.ones(rank)
    tensor = cp_to_tensor((weights, factors))
    _ = parafac(
        tensor, rank=rank, init="random", n_iter_max=1, random_state=random_state
    )


def test_sparse_parafac_svd_init():
    """Test that sparse parafac works with SVD initialization (uses argmax)."""
    tensor = sparse.random((5, 6, 7), density=0.3, random_state=1234)
    weights, factors = parafac(tensor, rank=2, init="svd", n_iter_max=3)
    assert [f.shape for f in factors] == [(5, 2), (6, 2), (7, 2)]


def test_sparse_argmax():
    x = sparse.random((5, 3), density=0.5, random_state=1)
    expected = np.argmax(x.todense(), axis=0)
    with sparse_context():
        result = tl.argmax(x, axis=0)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


def test_sparse_robust_pca():
    """Test that robust PCA stays sparse throughout its update loop."""
    tensor = sparse.random((4, 5, 6), density=0.2, random_state=1234)
    low_rank, sparse_error = robust_pca(tensor, n_iter_max=3, verbose=0)

    assert isinstance(low_rank, sparse.SparseArray)
    assert isinstance(sparse_error, sparse.SparseArray)
    assert low_rank.shape == tensor.shape
    assert sparse_error.shape == tensor.shape
    assert np.isfinite(low_rank.data).all()
    assert np.isfinite(sparse_error.data).all()
