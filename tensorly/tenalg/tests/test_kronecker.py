import numpy as np
from scipy.linalg import inv
from numpy.testing import assert_array_equal, assert_array_almost_equal
from .._kronecker import kronecker
from .._kronecker import inv_squared_kronecker
from .._khatri_rao import khatri_rao

# Author: Jean Kossaifi


def test_kronecker():
    """Test for kronecker product"""
    # Mathematical test
    a = np.array([[1, 2, 3], [3, 2, 1]])
    b = np.array([[2, 1], [2, 3]])
    true_res = np.array([[2, 1, 4, 2, 6, 3],
                         [2, 3, 4, 6, 6, 9],
                         [6, 3, 4, 2, 2, 1],
                         [6, 9, 4, 6, 2, 3]])
    res = kronecker([a, b])
    assert_array_equal(true_res, res)

    # Another test
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[0, 5], [6, 7]])
    true_res = np.array([[0, 5, 0, 10],
                         [6, 7, 12, 14],
                         [0, 15, 0, 20],
                         [18, 21, 24, 28]])
    res = kronecker([a, b])
    assert_array_equal(true_res, res)
    # Adding a third matrices
    c = np.array([[0, 1], [2, 0]])
    res = kronecker([c, a, b])
    assert(res.shape == (a.shape[0]*b.shape[0]*c.shape[0], a.shape[1]*b.shape[1]*c.shape[1]))
    assert_array_equal(res[:4, :4], c[0, 0]*true_res)
    assert_array_equal(res[:4, 4:], c[0, 1]*true_res)
    assert_array_equal(res[4:, :4], c[1, 0]*true_res)
    assert_array_equal(res[4:, 4:], c[1, 1]*true_res)


    # Test for the reverse argument
    matrix_list = [a, b]
    res = kronecker(matrix_list)
    assert_array_equal(res[:2, :2], a[0, 0]*b)
    assert_array_equal(res[:2, 2:], a[0, 1]*b)
    assert_array_equal(res[2:, :2], a[1, 0]*b)
    assert_array_equal(res[2:, 2:], a[1, 1]*b)
    # Check that the original list has not been reversed
    assert_array_equal(matrix_list[0], a)
    assert_array_equal(matrix_list[1], b)

    # Check the returned shape
    shapes = [[2, 3], [4, 5], [6, 7]]
    W = [np.random.randn(*shape) for shape in shapes]
    res = kronecker(W)
    assert (res.shape == (48, 105))

    # Khatri-rao is a column-wise kronecker product
    shapes = [[2, 1], [4, 1], [6, 1]]
    W = [np.random.randn(*shape) for shape in shapes]
    res = kronecker(W)
    assert (res.shape == (48, 1))

    # Khatri-rao product is a column-wise kronecker product
    kr = khatri_rao(W)
    for i, shape in enumerate(shapes):
        assert_array_equal(res, kr)

    a = np.array([[1, 2],
                  [0, 3]])
    b = np.array([[0.5, 1],
                  [1, 2]])
    true_res = np.array([[0.5, 1., 1., 2.],
                         [1., 2., 2., 4.],
                         [0., 0., 1.5, 3.],
                         [0., 0., 3., 6.]])
    assert_array_equal(kronecker([a, b]),  true_res)
    reversed_res = np.array([[ 0.5,  1. ,  1. ,  2. ],
                             [ 0. ,  1.5,  0. ,  3. ],
                             [ 1. ,  2. ,  2. ,  4. ],
                             [ 0. ,  3. ,  0. ,  6. ]])
    assert_array_equal(kronecker([a, b], reverse=True),  reversed_res)

    # Test while skipping a matrix
    shapes = [[2, 3], [4, 5], [6, 7]]
    U = [np.random.randn(*shape) for shape in shapes]
    res_1 = kronecker(U, skip_matrix=1)
    res_2 = kronecker([U[0]] + U[2:])
    assert_array_equal(res_1, res_2)

    res_1 = kronecker(U, skip_matrix=0)
    res_2 = kronecker(U[1:])
    assert_array_equal(res_1, res_2)



def test_inv_squared_kronecker():
    """Test for inv_squared_kronecker"""
    # Generate random matrices
    n_identity = 5
    mu = 10e-3
    dims = [[10, 10], [2, 3], [1, 20]]
    W = [np.random.random(dim) for dim in dims]

    # Easy-way to compute the inverse
    kron_W = kronecker(W)
    true_res = inv(np.dot(kron_W.T, kron_W) + mu*np.eye(kron_W.shape[1])*n_identity)*mu
    # Faster version
    res = inv_squared_kronecker(W, n_identity=n_identity, mu=mu)
    assert_array_almost_equal(true_res, res)

    true_res = inv(np.dot(kron_W.T, kron_W) + np.eye(kron_W.shape[1])*n_identity)
    # Faster version
    res = inv_squared_kronecker(W, n_identity=n_identity)
    assert_array_almost_equal(true_res, res)