import numpy as np
import tensorly as tl

from tensorly.solvers.nnls import (
    hals_nnls,
    fista,
    active_set_nnls,
)
from tensorly.testing import assert_, assert_array_equal, assert_array_almost_equal
from tensorly import tensor_to_vec, truncated_svd
import pytest

# Author: Jean Kossaifi
skip_tensorflow = pytest.mark.skipif(
    (tl.get_backend() == "tensorflow"),
    reason=f"Indexing with list not supported in TensorFlow",
)


def test_hals_nnls():
    """Test for hals_nnls operator. A linear system is Ax=b with known A, b and ground truth x is solved using HALS NNLS.
    We test that the estimated x_hals is almost equal to the ground truth x."""
    a = tl.tensor(np.random.rand(20, 10))
    true_res = tl.tensor(np.random.rand(10, 1))
    b = tl.dot(a, true_res)
    atb = tl.dot(tl.transpose(a), b)
    ata = tl.dot(tl.transpose(a), a)
    x_hals = hals_nnls(atb, ata)
    assert_array_almost_equal(true_res, x_hals, decimal=2)


def test_fista():
    """Test for fista operator. A linear system is Ax=b with known A, b and ground truth x is solved using FISTA.
    We test that the estimated x_fista is almost equal to the ground truth x."""
    a = tl.tensor(np.random.rand(20, 10))
    true_res = tl.tensor(np.random.rand(10, 1))
    b = tl.dot(a, true_res)
    atb = tl.dot(tl.transpose(a), b)
    ata = tl.dot(tl.transpose(a), a)
    x_fista = fista(atb, ata, tol=10e-16, n_iter_max=5000)
    assert_array_almost_equal(true_res, x_fista, decimal=2)


@skip_tensorflow
def test_active_set_nnls():
    """Test for active_set_nnls operator. A linear system is Ax=b with known A, b and ground truth x is solved using Active Set.
    We test that the estimated x_as is almost equal to the ground truth x"""
    a = tl.tensor(np.random.rand(20, 10))
    true_res = tl.tensor(np.random.rand(10, 1))
    b = tl.dot(a, true_res)
    atb = tl.dot(tl.transpose(a), b)
    ata = tl.dot(tl.transpose(a), a)
    x_as = active_set_nnls(tensor_to_vec(atb), ata)
    x_as = tl.reshape(x_as, tl.shape(atb))
    assert_array_almost_equal(true_res, x_as, decimal=2)


@pytest.mark.parametrize("tensor_equation", [False, True])
def test_fista_does_not_stop_when_coordinate_updates_cancel(tensor_equation):
    gram = tl.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    target = tl.tensor([[1.0], [7.0], [6.0]])
    initial = tl.tensor([[3.0], [2.0], [2.0]])
    expected = tl.tensor([[1.0], [3.5], [1.5]])
    kwargs = {}
    if tensor_equation:
        gram = [gram, tl.ones((1, 1))]
        kwargs["lr"] = 0.25

    result = fista(target, gram, x=initial, **kwargs)

    # The second update has opposite nonzero entries summing to zero.
    assert_array_almost_equal(result, expected, decimal=5)
