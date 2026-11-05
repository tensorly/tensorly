from ..decomposition import parafac, robust_pca
from ..tenalg import multi_mode_dot
from ..cp_tensor import cp_to_tensor
from .... import backend as tl

import pytest

if not tl.get_backend() == "numpy":
    pytest.skip("Tests for sparse only with numpy backend", allow_module_level=True)
pytest.importorskip("sparse")

import sparse
import numpy as np


@pytest.mark.parametrize("l2_reg", [0, 0.1])
def test_sparse_parafac(l2_reg):
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
        tensor,
        rank=rank,
        init="random",
        n_iter_max=1,
        random_state=random_state,
        l2_reg=l2_reg,
    )


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


@pytest.mark.parametrize("fill_value", [0, 2])
def test_sparse_fill_diagonal(fill_value):
    from tensorly.contrib.sparse import fill_diagonal

    data = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    tensor = sparse.COO.from_numpy(data, fill_value=fill_value)
    result = fill_diagonal(tensor, np.array([4.0, 5.0]))
    expected = data.copy()
    np.fill_diagonal(expected, [4.0, 5.0])
    assert isinstance(result, sparse.COO)
    np.testing.assert_array_equal(result.todense(), expected)
    np.testing.assert_array_equal(tensor.todense(), data)
    assert result.fill_value == fill_value
