"""Sparse-specific tests for the :mod:`tensorly.tenalg` module.
"""
from  .... import backend as tl

import pytest
import numpy as np
if not tl.get_backend() == "numpy":
    pytest.skip("Tests for sparse only with numpy backend", allow_module_level=True)
pytest.importorskip("sparse")

import tensorly.contrib.sparse as stl
from tensorly.contrib.sparse.kruskal_tensor import unfolding_dot_khatri_rao as sparse_unfolding_dot_khatri_rao

def test_sparse_unfolding_times_kruskal():
    """Test for unfolding_times_kruskal with sparse tensors
    
    We have already checked correctness in main backend
    Here, we check it is sparse-safe:
    the following example would blow-up memory if not sparse safe.
    """
    import sparse

    shape = (100, 101, 102, 100, 100, 100)
    rank = 5
    factors = [sparse.random((i, rank)) for i in shape]
    weights = np.ones(rank)
    tensor = stl.kruskal_to_tensor((weights, factors))
    
    for mode in range(tl.ndim(tensor)):
        # Will blow-up memory if not sparse-safe
        _ = sparse_unfolding_dot_khatri_rao(tensor, (weights, factors), mode)