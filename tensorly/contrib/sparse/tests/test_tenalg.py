"""Sparse-specific tests for the :mod:`tensorly.tenalg` module."""

from .... import backend as tl

import pytest
import numpy as np

if not tl.get_backend() == "numpy":
    pytest.skip("Tests for sparse only with numpy backend", allow_module_level=True)
pytest.importorskip("sparse")

import tensorly.contrib.sparse as stl
from tensorly.contrib.sparse.backend import sparse_context
from tensorly.contrib.sparse.cp_tensor import (
    unfolding_dot_khatri_rao as sparse_unfolding_dot_khatri_rao,
)
from tensorly.tenalg.svd import truncated_svd


def test_sparse_unfolding_times_cp():
    """Test for unfolding_times_cp with sparse tensors

    We have already checked correctness in main backend
    Here, we check it is sparse-safe:
    the following example would blow-up memory if not sparse safe.
    """
    import sparse

    shape = (300, 300, 300, 10)
    rank = 5
    factors = [sparse.random((i, rank), density=0.08) for i in shape]
    weights = np.ones(rank)
    tensor = stl.cp_to_tensor((weights, factors))

    for mode in range(tl.ndim(tensor)):
        # Will blow-up memory if not sparse-safe
        _ = sparse_unfolding_dot_khatri_rao(tensor, (weights, factors), mode)


@pytest.mark.parametrize("shape", [(4, 30), (30, 4)])
def test_sparse_truncated_svd_with_rank_deficient_matrix(shape):
    """Test that the sparse backend provides a finite reduced SVD."""
    import sparse

    matrix = sparse.COO.from_numpy(np.ones(shape))

    with sparse_context():
        U, S, V = truncated_svd(matrix, n_eigenvecs=min(matrix.shape))

    U = U.todense() if hasattr(U, "todense") else np.asarray(U)
    V = V.todense() if hasattr(V, "todense") else np.asarray(V)
    assert np.isfinite(U).all()
    assert np.isfinite(S).all()
    assert np.isfinite(V).all()
    np.testing.assert_allclose(U @ (S[:, None] * V), matrix.todense(), atol=1e-6)
    np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-6)
    np.testing.assert_allclose(V @ V.T, np.eye(V.shape[0]), atol=1e-6)


def test_sparse_partial_unfold():
    """partial_unfold works on sparse tensors and matches the dense result"""
    import sparse
    from tensorly.base import partial_unfold

    dense = np.arange(24, dtype=float).reshape((2, 3, 2, 2))
    dense[dense % 3 == 0] = 0
    tensor = sparse.COO.from_numpy(dense)

    with sparse_context():
        for mode, skip_begin, skip_end in [(0, 1, 0), (1, 1, 0), (0, 2, 0), (0, 1, 1)]:
            res = partial_unfold(tensor, mode, skip_begin, skip_end)
            assert isinstance(res, sparse.COO)
            np.testing.assert_array_equal(
                res.todense(), partial_unfold(dense, mode, skip_begin, skip_end)
            )
