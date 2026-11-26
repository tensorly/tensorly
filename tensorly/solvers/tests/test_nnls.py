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
@pytest.mark.parametrize(
    "dtype, offset", [("float32", 2.0**-13), ("float64", 2.0**-27)]
)
def test_active_set_nnls_singular_update(dtype, offset):
    a = tl.tensor([[1.0, 1.0, 1.0], [0.0, offset, 1.0]], dtype=getattr(tl, dtype))
    b = tl.tensor([1.0, offset - 1.0], **tl.context(a))
    # Dropping the third variable leaves a singular passive block.
    x = active_set_nnls(
        tl.dot(tl.transpose(a), b),
        tl.dot(tl.transpose(a), a),
        x=tl.ones(3, **tl.context(a)),
    )
    assert_array_almost_equal(x, tl.tensor([1.0, 0.0, 0.0], **tl.context(a)))


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
