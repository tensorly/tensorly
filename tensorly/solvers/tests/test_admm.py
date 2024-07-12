import numpy as np
import tensorly as tl

from tensorly.solvers.admm import admm
from tensorly.testing import assert_, assert_array_equal, assert_array_almost_equal
from tensorly import tensor_to_vec, truncated_svd
import pytest

# Author: Jean Kossaifi
skip_tensorflow = pytest.mark.skipif(
    (tl.get_backend() == "tensorflow"),
    reason=f"Indexing with list not supported in TensorFlow",
)


def test_admm():
    """Test for admm operator. A linear system Ax=b with known A, b and known ground truth x is solved with ADMM, which outputs an estimate x_admm. This test checks if x_admm is almost the true x."""
    a = tl.tensor(np.random.rand(20, 10))
    true_res = tl.tensor(np.random.rand(10, 10))
    b = tl.dot(a, true_res)
    atb = tl.dot(tl.transpose(a), b)
    ata = tl.dot(tl.transpose(a), a)
    dual = tl.zeros(tl.shape(atb))
    x_init = tl.zeros(tl.shape(atb))
    x_admm, _, _ = admm(tl.transpose(atb), tl.transpose(ata), x=x_init, dual_var=dual)
    assert_array_almost_equal(true_res, tl.transpose(x_admm), decimal=2)
