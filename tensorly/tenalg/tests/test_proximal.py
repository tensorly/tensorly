import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..proximal import svd_thresholing, soft_thresholding, inplace_soft_thresholding
from ..proximal import procrustes

# Author: Jean Kossaifi


def test_soft_thresholding():
    """Test for shrinkage"""

    # small test
    tensor = np.array([[[1, 2, 3], [4.3, -1.2, 3]],
                       [[0.5, -5, -1.3], [1.2, 3.7, -9]],
                       [[-2, 0, 1.0], [0.5, -0.5, 1.1]]], dtype=np.float64)
    threshold = 1.1
    copy_tensor = np.copy(tensor)
    res = soft_thresholding(tensor, threshold)
    true_res = np.array([[[0, 0.9, 1.9], [3.2, -0.1, 1.9]],
                         [[0, -3.9, -0.2], [0.1, 2.6, -7.9]],
                         [[-0.9, 0, 0], [0, -0, 0]]], dtype=np.float64)
    # account for floating point errors: np array have a precision of around 2e-15
    # check np.finfo(np.float64).eps
    assert_array_almost_equal(true_res, res, decimal=15)
    # Check that we did not change the original tensor
    assert_array_equal(copy_tensor, tensor)

    # Another test
    tensor = np.array([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.02, -3.4]])
    copy_tensor = np.copy(tensor)
    threshold = 1.1
    true_res = np.array([[0, 0.9, 0.4], [2.9, -4.9, 0], [0, 0, -2.3]])
    res = soft_thresholding(tensor, threshold)
    assert_array_almost_equal(true_res, res, decimal=15)
    assert_array_equal(copy_tensor, tensor)

    # Test with missing values
    tensor = np.array([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.02, -3.4]])
    copy_tensor = np.copy(tensor)
    mask = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    threshold = 1.1*mask
    true_res = np.array([[1, 0.9, 0.4], [2.9, -6, 0], [0, 0, -3.4]])
    res = soft_thresholding(tensor, threshold)
    assert_array_almost_equal(true_res, res, decimal=15)
    assert_array_equal(copy_tensor, tensor)


def test_inplace_soft_thresholding():
    """Test for inplace_shrinkage

    Notes
    -----
    Assumes that shrinkage is tested and works as expected
    """
    shape = (4, 5, 3, 2)
    tensor = np.random.random(shape)
    threshold = 0.21
    true_res = soft_thresholding(tensor, threshold)
    res = inplace_soft_thresholding(tensor, threshold)
    assert_array_almost_equal(true_res, res)

    # Check that the soft thresholding was done inplace
    assert (res is tensor)


def test_svd_thresholing():
    """Test for singular_value_thresholding operator"""
    U = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    singular_values = [0.4, 2.1, -2]
    tensor = U.dot(np.diag(singular_values).dot(U.T))
    shrinked_singular_values = [0, 1.6, -1.5]
    true_res = U.dot(np.diag(shrinked_singular_values).dot(U.T))
    res = svd_thresholing(tensor, 0.5)
    assert_array_almost_equal(true_res, res)


def test_procrustes():
    """Test for procrustes operator"""
    U = np.random.rand(20, 10)
    S, _, V = np.linalg.svd(U, full_matrices=False)
    true_res = S.dot(V)
    res = procrustes(U)
    assert_array_almost_equal(true_res, res)

