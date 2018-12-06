import numpy as np

from ... import backend as T
from .._kronecker import kronecker
from .._khatri_rao import khatri_rao
from ...testing import assert_array_equal, assert_array_almost_equal

# Author: Jean Kossaifi


def test_kronecker():
    """Test for kronecker product"""
    # Mathematical test
    a = T.tensor([[1, 2, 3], [3, 2, 1]])
    b = T.tensor([[2, 1], [2, 3]])
    true_res = T.tensor([[2, 1, 4, 2, 6, 3],
                         [2, 3, 4, 6, 6, 9],
                         [6, 3, 4, 2, 2, 1],
                         [6, 9, 4, 6, 2, 3]])
    res = kronecker([a, b])
    assert_array_equal(true_res, res)

    # Another test
    a = T.tensor([[1, 2], [3, 4]])
    b = T.tensor([[0, 5], [6, 7]])
    true_res = T.tensor([[0, 5, 0, 10],
                         [6, 7, 12, 14],
                         [0, 15, 0, 20],
                         [18, 21, 24, 28]])
    res = kronecker([a, b])
    assert_array_equal(true_res, res)
    # Adding a third matrices
    c = T.tensor([[0, 1], [2, 0]])
    res = kronecker([c, a, b])
    assert (res.shape == (a.shape[0]*b.shape[0]*c.shape[0], a.shape[1]*b.shape[1]*c.shape[1]))
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
    W = [T.tensor(np.random.randn(*shape)) for shape in shapes]
    res = kronecker(W)
    assert (res.shape == (48, 105))

    # Khatri-rao is a column-wise kronecker product
    shapes = [[2, 1], [4, 1], [6, 1]]
    W = [T.tensor(np.random.randn(*shape)) for shape in shapes]
    res = kronecker(W)
    assert (res.shape == (48, 1))

    # Khatri-rao product is a column-wise kronecker product
    kr = khatri_rao(W)
    for i, shape in enumerate(shapes):
        assert_array_almost_equal(res, kr)

    a = T.tensor([[1, 2],
                  [0, 3]])
    b = T.tensor([[0.5, 1],
                  [1, 2]])
    true_res = T.tensor([[0.5, 1., 1., 2.],
                         [1., 2., 2., 4.],
                         [0., 0., 1.5, 3.],
                         [0., 0., 3., 6.]])
    assert_array_equal(kronecker([a, b]),  true_res)
    reversed_res = T.tensor([[ 0.5,  1. ,  1. ,  2. ],
                             [ 0. ,  1.5,  0. ,  3. ],
                             [ 1. ,  2. ,  2. ,  4. ],
                             [ 0. ,  3. ,  0. ,  6. ]])
    assert_array_equal(kronecker([a, b], reverse=True),  reversed_res)

    # Test while skipping a matrix
    shapes = [[2, 3], [4, 5], [6, 7]]
    U = [T.tensor(np.random.randn(*shape)) for shape in shapes]
    res_1 = kronecker(U, skip_matrix=1)
    res_2 = kronecker([U[0]] + U[2:])
    assert_array_equal(res_1, res_2)

    res_1 = kronecker(U, skip_matrix=0)
    res_2 = kronecker(U[1:])
    assert_array_equal(res_1, res_2)
