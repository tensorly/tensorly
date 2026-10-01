import tensorly as tl
import numpy as np

from ...testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_,
)
from ... import random
from .. import tensordot
import pytest


def test_batched_tensordot():
    shape = [3, 4, 2]
    vecs = [random.random_tensor(s) for s in shape]
    tensor = random.random_tensor(shape)

    # Equivalence with inner product when contracting with self along all modes
    res = tensordot(tensor, tensor, modes=3)  # [[0, 1, 2], [0, 1, 2]])
    true_res = tl.tenalg.inner(tensor, tensor, n_modes=3)
    assert_array_almost_equal(true_res, res, decimal=5)
    # Equivalent to the above expression
    res = tensordot(tensor, tensor, modes=[[0, 1, 2], [0, 1, 2]])
    assert_array_almost_equal(true_res, res, decimal=5)

    # Equivalence with n-mode-dot
    for mode, vec in enumerate(vecs):
        res = tensordot(tensor, vec, (mode, 0))
        true_res = tl.tenalg.mode_dot(tensor, vec, mode)
        assert_array_almost_equal(true_res, res, decimal=5)

    # Multi-mode-dot
    res = tensordot(tensordot(tensor, vecs[0], (0, 0)), vecs[1], (0, 0))
    true_res = tl.tenalg.multi_mode_dot(tensor, vecs[:2], [0, 1])

    # Wrong number of modes
    with assert_raises(ValueError):
        tensordot(tensor, tensor, modes=[[0, 2], [0, 1, 2]])

    # size mismatch
    with assert_raises(ValueError):
        tensordot(tensor, vecs[1], modes=(0, 0))

    # Test Batched tensor dot
    tensor = random.random_tensor((4, 2, 3, 3))
    tensor2 = random.random_tensor((3, 4, 2, 3))
    res = tensordot(tensor, tensor2, ((0, 3), (1, 3)), batched_modes=(1, 2))
    # Check for each sample of the batch-size individually
    for i in range(2):
        true_res = tl.tensordot(tensor[:, i], tensor2[:, :, i], ((0, 2), (1, 2)))
        assert_array_almost_equal(res[i], true_res, decimal=5)

    # Test for actual tensordot
    tensor = random.random_tensor((4, 3, 3))
    tensor2 = random.random_tensor((3, 4, 2))
    res = tensordot(tensor, tensor2, modes=(), batched_modes=())
    assert_(tuple(res.shape) == (4, 3, 3, 3, 4, 2))
    res = tensordot(tensor, tensor2, modes=(), batched_modes=((0,), (1,)))
    assert_(tuple(res.shape) == (4, 3, 3, 3, 2))


@pytest.mark.parametrize("tenalg_backend", ["core", "einsum"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("batch_shape", [(2, 3), (3, 3), (1, 3)])
@pytest.mark.parametrize(
    "layout", ["matmul", "interleaved", "dot", "outer", "three_batches"]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_batched_tensordot_axis_order(
    tenalg_backend, dtype, batch_shape, layout, reverse
):
    """Keep batch entries aligned when paired batch axes have different orders.

    Regression for gh-637: the core implementation reshaped batch dimensions
    in tensor-axis order after transposing them in the supplied batch order,
    mixing values between independent batches even when the shape was correct.
    NumPy einsum provides an independent reference for the axis pairing across
    contraction layouts, reversed batch lists, and equal or unequal batch sizes.
    """
    a, b = batch_shape
    if layout == "matmul":
        shape1, shape2 = (a, b, 3, 4), (b, a, 4, 5)
        modes, batch1, batch2 = (3, 2), [0, 1], [1, 0]
        equation = "abic,bacj->abij"
    elif layout == "interleaved":
        shape1, shape2 = (3, a, 4, b), (b, 4, a, 5)
        modes, batch1, batch2 = (2, 1), [1, 3], [2, 0]
        equation = "iacb,bcaj->iabj"
    elif layout == "dot":
        shape1, shape2 = (a, b, 4), (b, a, 4)
        modes, batch1, batch2 = (2, 2), [0, 1], [1, 0]
        equation = "abc,bac->ab"
    elif layout == "outer":
        shape1, shape2 = (a, b, 3), (b, a, 5)
        modes, batch1, batch2 = (), [0, 1], [1, 0]
        equation = "abi,baj->abij"
    else:
        shape1, shape2 = (a, 2, 3, b, 4), (b, a, 4, 2, 5)
        modes, batch1, batch2 = (4, 2), [0, 1, 3], [1, 3, 0]
        equation = "adibc,bacdj->adibj"
    if reverse:
        batch1, batch2 = batch1[::-1], batch2[::-1]
    values1 = (np.arange(np.prod(shape1)).reshape(shape1) / 64).astype(dtype)
    values2 = (np.arange(np.prod(shape2)).reshape(shape2) / 128 + 0.5).astype(dtype)
    expected = np.einsum(equation, values1, values2)
    tensor1 = tl.tensor(values1, dtype=getattr(tl, dtype))
    tensor2 = tl.tensor(values2, dtype=getattr(tl, dtype))
    original_backend = tl.tenalg.get_backend()
    try:
        tl.tenalg.set_backend(tenalg_backend)
        actual = tensordot(
            tensor1, tensor2, modes=modes, batched_modes=(batch1, batch2)
        )
    finally:
        tl.tenalg.set_backend(original_backend)
    assert_equal(tl.shape(actual), expected.shape)
    assert_equal(tl.context(actual)["dtype"], tl.context(tensor1)["dtype"])
    assert_array_almost_equal(actual, expected)
    assert_array_equal(tensor1, values1)
    assert_array_equal(tensor2, values2)
