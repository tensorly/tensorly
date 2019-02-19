import tensorly as tl

from tensorly.contrib.sparse.decomposition import partial_tucker

import pytest

pytestmark = pytest.mark.skipif(tl.get_backend() != "sparse",
                                        reason="Operation only supported in Sparse")

try:
    import sparse
    import numpy as np
except ImportError:
    pass

def test_sparse_backend():
    tensor = tl.tensor([[1, 2], [3, 4]])
    assert isinstance(tensor, sparse.COO)

    tensor2 = tl.dot(tensor, tl.tensor([[-1], [2]]))
    assert isinstance(tensor2, sparse.COO)
    np.testing.assert_equal(tensor2, tl.tensor([[3], [5]]))

def test_sparse_partial_tucker():
    # Make sure the algorithm stays sparse. This will run out of memory on
    # most machines if the algorithm densifies.
    random_state = 1234
    rank = 3
    core = sparse.random((2482, rank, rank), random_state=random_state)
    factors = [sparse.random((2862, rank), random_state=random_state),
               sparse.random((14036, rank), random_state=random_state)]

    tensor = tl.tenalg.multi_mode_dot(core, factors, [1, 2])
    new_core, new_factors = partial_tucker(tensor, modes=[1, 2], rank=[rank, rank],
        init='random', tol=1e-3, random_state=random_state)
