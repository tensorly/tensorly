import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises, assert_equal
from ...base import fold, unfold
from .._kronecker import kronecker
from ..n_mode_product import mode_dot, multi_mode_dot


def test_mode_dot():
    """Test for mode_dot (n_mode_product)"""
    X = np.array([[[1, 13],
                   [4, 16],
                   [7, 19],
                   [10, 22]],

                  [[2, 14],
                   [5, 17],
                   [8, 20],
                   [11, 23]],

                  [[3, 15],
                   [6, 18],
                   [9, 21],
                   [12, 24]]])
    # tensor times matrix
    U = np.array([[1, 3, 5],
                  [2, 4, 6]])
    true_res = np.array([[[22, 130],
                          [49, 157],
                          [76, 184],
                          [103, 211]],
                         [[28, 172],
                          [64, 208],
                          [100, 244],
                          [136, 280]]])
    res = mode_dot(X, U, 0)
    assert_array_equal(true_res, res)

    #######################
    # tensor times vector #
    #######################
    U = np.array([1, 2, 3, 4])
    true_res = np.array([[70, 190],
                         [80, 200],
                         [90, 210]])
    res = mode_dot(X, U, 1)
    assert_array_equal(true_res, res)

    # Using equivalence with unfolded expression
    X = np.random.random((2, 4, 5))
    U = np.random.random((3, 4))
    true_res = fold(np.dot(U, unfold(X, 1)), 1, (2, 3, 5))
    res = mode_dot(X, U, 1)
    assert (res.shape == (2, 3, 5))
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
    X = np.random.random((2, 4, 5))
    U = np.random.random((3, 4))
    res = unfold(mode_dot(X, U, 1), 1)
    assert_array_almost_equal(U.dot(unfold(X, 1)), res)


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
    X = np.array([[1, 2],
                  [0, -1]])
    U = [np.array([2, 1]),
         np.array([-1, 1])]
    true_res = np.array([1])
    res = multi_mode_dot(X, U, [0, 1])
    assert_array_equal(true_res, res)

    X = np.arange(12).reshape((3, 4))
    U = np.random.random((3, 5))
    res_1 = multi_mode_dot(X, [U], modes=[0], transpose=True)
    res_2 = np.dot(U.T, X)
    assert_array_almost_equal(res_1, res_2)

    dims = [2, 3, 4, 5]
    X = np.random.randn(*dims)
    factors = [np.random.rand(dims[i], X.shape[i]) for i in range(X.ndim)]
    true_res = factors[0].dot(unfold(X, 0)).dot(kronecker(factors[1:]).T)
    n_mode_res = multi_mode_dot(X, factors)
    assert_array_almost_equal(true_res, unfold(n_mode_res, 0))
    for i in range(X.ndim):
        indices = [j for j in range(X.ndim) if j != i]
        sub_factors = [factors[j] for j in indices]
        true_res = factors[i].dot(unfold(X, i)).dot(kronecker(sub_factors).T)
        res = unfold(n_mode_res, i)
        temp = multi_mode_dot(X, sub_factors, modes=indices)
        res2 = factors[i].dot(unfold(temp, i))
        assert_equal(true_res.shape, res.shape, err_msg='shape should be {}, is {}'.format(true_res.shape, res.shape))
        assert_array_almost_equal(true_res, res)
        assert_array_almost_equal(true_res, res2)

    # Test skipping a factor
    dims = [2, 3, 4, 5]
    X = np.random.randn(*dims)
    factors = [np.random.rand(dims[i], X.shape[i]) for i in range(X.ndim)]
    res_1 = multi_mode_dot(X, factors, skip=1)
    res_2 = multi_mode_dot(X, [factors[0]] + factors[2:], modes=[0, 2, 3])
    assert_array_equal(res_1, res_2)


