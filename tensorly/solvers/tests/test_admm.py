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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", [None, "cuda"])
def test_admm_preserves_context(dtype, device):
    context = {"dtype": getattr(tl, dtype)}
    if device is not None:
        if tl.get_backend() != "pytorch":
            pytest.skip("Explicit CUDA device selection requires the PyTorch backend")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        context["device"] = device

    # A diagonal quadratic has a known nonnegative solution: clip each
    # unconstrained coordinate independently at zero.
    gram = tl.tensor([[1, 0, 0], [0, 4, 0], [0, 0, 9]], **context)
    unconstrained = tl.tensor([[-1, 2, 3], [4, -5, 6]], **context)
    rhs = tl.dot(unconstrained, gram)
    result, split, dual = admm(
        rhs,
        gram,
        x=tl.zeros(tl.shape(rhs), **context),
        dual_var=tl.zeros(tl.shape(rhs), **context),
        non_negative=True,
        n_const=1,
        order=0,
        n_iter_max=200,
        tol=1e-8,
    )
    assert_array_almost_equal(result, tl.clip(unconstrained, 0, None), decimal=4)
    for value in (result, split, dual):
        assert tl.context(value) == tl.context(gram)
