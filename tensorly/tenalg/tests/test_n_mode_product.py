import itertools

import numpy as np
import pytest

from ... import backend as T
from ...testing import (
    assert_array_equal,
    assert_equal,
    assert_array_almost_equal,
    assert_raises,
)
from ...base import fold, unfold


from .. import backend_context, kronecker
from .. import mode_dot, multi_mode_dot


def test_mode_dot():
    """Test for mode_dot (n_mode_product)"""
    X = T.tensor(
        [
            [[1, 13], [4, 16], [7, 19], [10, 22]],
            [[2, 14], [5, 17], [8, 20], [11, 23]],
            [[3, 15], [6, 18], [9, 21], [12, 24]],
        ]
    )
    # tensor times matrix
    U = T.tensor([[1, 3, 5], [2, 4, 6]])
    true_res = T.tensor(
        [
            [[22, 130], [49, 157], [76, 184], [103, 211]],
            [[28, 172], [64, 208], [100, 244], [136, 280]],
        ]
    )
    res = mode_dot(X, U, 0)
    assert_array_equal(true_res, res)

    #######################
    # tensor times vector #
    #######################
    # Test with a matrix
    U = T.tensor([1, 2, 3, 4])
    true_res = T.tensor([[70, 190], [80, 200], [90, 210]])
    res = mode_dot(X, U, 1)
    assert_array_equal(true_res, res)
    # Test with a third order tensor
    X = T.tensor(np.arange(24, dtype=float).reshape((3, 4, 2)))
    v = T.tensor(np.arange(4, dtype=float))
    true_res = [[28, 34], [76, 82], [124, 130]]
    res = mode_dot(X, v, 1)
    assert_array_equal(true_res, res)

    # Using equivalence with unfolded expression
    X = T.tensor(np.random.random((2, 4, 5)))
    U = T.tensor(np.random.random((3, 4)))
    true_res = fold(T.dot(U, unfold(X, 1)), 1, (2, 3, 5))
    res = mode_dot(X, U, 1)
    assert tuple(res.shape) == (2, 3, 5)
    assert_array_almost_equal(true_res, res)

    #########################################
    # Test for errors that should be raised #
    #########################################
    with assert_raises(ValueError):
        mode_dot(X, U, 0)
    # Same test for the vector case
    with assert_raises(ValueError):
        mode_dot(X, U[:, 0], 0)
    # Cannot take mode product of tensor with tensor
    with assert_raises(ValueError):
        mode_dot(X, X, 0)

    # Test using the equivalence with unfolded expression
    X = T.tensor(np.random.random((2, 4, 5)))
    U = T.tensor(np.random.random((3, 4)))
    res = unfold(mode_dot(X, U, 1), 1)
    assert_array_almost_equal(T.dot(U, unfold(X, 1)), res)


def test_multi_mode_dot():
    """Test for multi_mode_dot

    Notes
    -----
    First a numerical test (ie compute by hand and check)
    Then use that the following expressions are equivalent:

    * X x_1 U_1 x ... x_n U_n
    * U_1 x unfold(X, 1) x kronecker(U_2, ..., U_n).T
    * U_1 x unfold(X x_2 U_2 x ... x_n U_n)
    """
    X = T.tensor([[1, 2], [0, -1]])
    U = [T.tensor([2, 1]), T.tensor([-1, 1])]
    true_res = T.tensor([1])
    res = multi_mode_dot(X, U, [0, 1])
    assert_array_equal(true_res, res)

    X = T.tensor(np.arange(12, dtype=float).reshape((3, 4)))
    U = T.tensor(np.random.random((3, 5)))
    res_1 = multi_mode_dot(X, [U], modes=[0], transpose=True)
    res_2 = T.dot(T.transpose(U), X)
    assert_array_almost_equal(res_1, res_2)

    dims = [2, 3, 4, 5]
    X = T.tensor(np.random.randn(*dims))
    factors = [T.tensor(np.random.rand(dims[i], X.shape[i])) for i in range(T.ndim(X))]
    true_res = T.dot(
        T.dot(factors[0], unfold(X, 0)), T.transpose(kronecker(factors[1:]))
    )
    n_mode_res = multi_mode_dot(X, factors)
    assert_array_almost_equal(true_res, unfold(n_mode_res, 0), decimal=5)
    for i in range(T.ndim(X)):
        indices = [j for j in range(T.ndim(X)) if j != i]
        sub_factors = [factors[j] for j in indices]
        true_res = T.dot(
            T.dot(factors[i], unfold(X, i)), T.transpose(kronecker(sub_factors))
        )
        res = unfold(n_mode_res, i)
        temp = multi_mode_dot(X, sub_factors, modes=indices)
        res2 = T.dot(factors[i], unfold(temp, i))
        assert_equal(
            true_res.shape,
            res.shape,
            err_msg=f"shape should be {true_res.shape}, is {res.shape}",
        )
        assert_array_almost_equal(true_res, res, decimal=5)
        assert_array_almost_equal(true_res, res2, decimal=5)

    # Test skipping a factor
    dims = [2, 3, 4, 5]
    X = T.tensor(np.random.randn(*dims))
    factors = [T.tensor(np.random.rand(dims[i], X.shape[i])) for i in range(T.ndim(X))]
    res_1 = multi_mode_dot(X, factors, skip=1)
    res_2 = multi_mode_dot(X, [factors[0]] + factors[2:], modes=[0, 2, 3])
    assert_array_equal(res_1, res_2)

    # Test contracting with a vector
    shape = (3, 5, 4, 2)
    X = T.ones(shape)
    vecs = [T.ones(s) for s in shape]
    res = multi_mode_dot(X, vecs)
    # result should be a scalar
    assert_equal(T.shape(res), ())
    assert_array_almost_equal(res, np.prod(shape))

    # Average pooling each mode
    # Order should not matter
    vecs = [vecs[i] / s for i, s in enumerate(shape)]
    for modes in itertools.permutations(range(len(shape))):
        res = multi_mode_dot(X, [vecs[i] for i in modes], modes=modes)
        assert_equal(T.shape(res), ())
        assert_array_almost_equal(res, 1)


@pytest.mark.parametrize("tenalg_backend", ["core", "einsum"])
@pytest.mark.parametrize("modes", [None, (0, 1, 2), (2, 0, 1), (2, 1, 0)])
@pytest.mark.parametrize("skip", [None, 0, 1, 2])
@pytest.mark.parametrize("transpose", [False, True])
def test_multi_mode_dot_skip_original_factor(tenalg_backend, modes, skip, transpose):
    values = np.arange(24, dtype=float).reshape(2, 3, 4) / 16
    factors = [
        (np.arange(4, dtype=float).reshape(2, 2) + 1) / 4,
        (np.arange(3, dtype=float) + 1) / 4,
        (np.arange(16, dtype=float).reshape(4, 4) + 1) / 8,
    ]
    ordered_modes = (0, 1, 2) if modes is None else modes
    skipped_mode = None if skip is None else ordered_modes[skip]
    expected = {
        None: np.einsum("abc,da,b,ec->de", values, *factors),
        0: np.einsum("abc,b,dc->ad", values, factors[1], factors[2]),
        1: np.einsum("abc,da,ec->dbe", values, factors[0], factors[2]),
        2: np.einsum("abc,da,b->dc", values, factors[0], factors[1]),
    }[skipped_mode]
    ordered_factors = [
        factors[mode].T if transpose else factors[mode] for mode in ordered_modes
    ]
    tensor = T.tensor(values, dtype=T.float64)
    tensor_factors = [T.tensor(factor, dtype=T.float64) for factor in ordered_factors]

    with backend_context(tenalg_backend):
        actual = multi_mode_dot(
            tensor, tensor_factors, modes=modes, skip=skip, transpose=transpose
        )

    assert_equal(T.shape(actual), expected.shape)
    assert_array_almost_equal(actual, expected)
    assert_array_equal(tensor, values)
    for factor, original in zip(tensor_factors, ordered_factors):
        assert_array_equal(factor, original)


@pytest.mark.parametrize("tenalg_backend", ["core", "einsum"])
@pytest.mark.parametrize("skip", [0, 1])
@pytest.mark.parametrize("transpose", [False, True])
def test_multi_mode_dot_skip_partial_modes(tenalg_backend, skip, transpose):
    values = np.arange(24, dtype=float).reshape(2, 3, 4) / 16
    factors = [
        (np.arange(20, dtype=float).reshape(5, 4) + 1) / 8,
        (np.arange(12, dtype=float).reshape(6, 2) + 1) / 4,
    ]
    if skip == 0:
        expected = np.einsum("abc,da->dbc", values, factors[1])
    else:
        expected = np.einsum("abc,dc->abd", values, factors[0])
    tensor = T.tensor(values, dtype=T.float64)
    tensor_factors = [
        T.tensor(factor.T if transpose else factor, dtype=T.float64)
        for factor in factors
    ]

    with backend_context(tenalg_backend):
        actual = multi_mode_dot(
            tensor, tensor_factors, modes=(2, 0), skip=skip, transpose=transpose
        )

    assert_equal(T.shape(actual), expected.shape)
    assert_array_almost_equal(actual, expected)


@pytest.mark.parametrize("tenalg_backend", ["core", "einsum"])
def test_multi_mode_dot_skip_only_factor(tenalg_backend):
    tensor = T.tensor(np.arange(24, dtype=float).reshape(2, 3, 4))
    with backend_context(tenalg_backend):
        actual = multi_mode_dot(tensor, [T.ones((5, 4))], modes=[2], skip=0)
    assert_array_equal(actual, tensor)
