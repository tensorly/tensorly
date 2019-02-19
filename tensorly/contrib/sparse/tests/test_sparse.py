import tensorly as tl

from tensorly.contrib.sparse.decomposition import partial_tucker

import pytest

pytestmark = pytest.mark.skipif(tl.get_backend() != "sparse",
                                        reason="Operation not supported in TensorFlow")

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
