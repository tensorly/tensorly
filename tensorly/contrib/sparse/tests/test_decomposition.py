from ..decomposition import parafac
from ..tenalg import multi_mode_dot
from ..kruskal_tensor import kruskal_to_tensor
from .... import backend as tl 

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
    factors = [sparse.random((2862, rank), random_state=random_state),
               sparse.random((14036, rank), random_state=random_state)]
    weights = np.ones(rank)
    tensor = kruskal_to_tensor((weights, factors))
    _ = parafac(tensor, rank=rank, init='random', 
                n_iter_max=1, random_state=random_state)
