import numpy as np

from ... import backend as T
from ..proximal import svd_thresholding, soft_thresholding
from ..proximal import procrustes
from ...testing import assert_array_equal, assert_array_almost_equal

# Author: Jean Kossaifi


def test_soft_thresholding():
    """Test for shrinkage"""
    # small test
    tensor = T.tensor([[[1, 2, 3], [4.3, -1.2, 3]],
                       [[0.5, -5, -1.3], [1.2, 3.7, -9]],
                       [[-2, 0, 1.0], [0.5, -0.5, 1.1]]])
    threshold = 1.1
    copy_tensor = T.copy(tensor)
    res = soft_thresholding(tensor, threshold)
    true_res = T.tensor([[[0, 0.9, 1.9], [3.2, -0.1, 1.9]],
                         [[0, -3.9, -0.2], [0.1, 2.6, -7.9]],
                         [[-0.9, 0, 0], [0, -0, 0]]])
    # account for floating point errors: np array have a precision of around 2e-15
    # check np.finfo(np.float64).eps
    assert_array_almost_equal(true_res, res)
    # Check that we did not change the original tensor
    assert_array_equal(copy_tensor, tensor)

    # Another test
    tensor = T.tensor([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.02, -3.4]])
    copy_tensor = T.copy(tensor)
    threshold = 1.1
    true_res = T.tensor([[0, 0.9, 0.4], [2.9, -4.9, 0], [0, 0, -2.3]])
    res = soft_thresholding(tensor, threshold)
    assert_array_almost_equal(true_res, res)
    assert_array_equal(copy_tensor, tensor)

    # Test with missing values
    tensor = T.tensor([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.02, -3.4]])
    copy_tensor = T.copy(tensor)
    mask = T.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    threshold = 1.1 * mask
    true_res = T.tensor([[1, 0.9, 0.4], [2.9, -6, 0], [0, 0, -3.4]])
    res = soft_thresholding(tensor, threshold)
    assert_array_almost_equal(true_res, res)
    assert_array_equal(copy_tensor, tensor)


def test_svd_thresholding():
    """Test for singular_value_thresholding operator"""
    U = T.tensor([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    singular_values = T.tensor([0.4, 2.1, -2])
    tensor = T.dot(U, T.reshape(singular_values, (-1, 1)) * T.transpose(U))
    shrinked_singular_values = T.tensor([0, 1.6, -1.5])
    true_res = T.dot(U, T.reshape(shrinked_singular_values, (-1, 1)) * T.transpose(U))
    res = svd_thresholding(tensor, 0.5)
    assert_array_almost_equal(true_res, res)


def test_procrustes():
    """Test for procrustes operator"""
    U = T.tensor(np.random.rand(20, 10))
    S, _, V = T.partial_svd(U, n_eigenvecs=min(U.shape))
    true_res = T.dot(S, V)
    res = procrustes(U)
    assert_array_almost_equal(true_res, res)
