import numpy as np
import tensorly as tl

from tensorly.tenalg.proximal import (
    svd_thresholding,
    smoothness_prox,
    soft_thresholding,
    procrustes,
    hard_thresholding,
    soft_sparsity_prox,
    simplex_prox,
    normalized_sparsity_prox,
    monotonicity_prox,
    unimodality_prox,
    l2_prox,
    l2_square_prox,
)
from tensorly.testing import assert_, assert_array_equal, assert_array_almost_equal
from tensorly import truncated_svd
import pytest

# Author: Jean Kossaifi
skip_tensorflow = pytest.mark.skipif(
    (tl.get_backend() == "tensorflow"),
    reason=f"Indexing with list not supported in TensorFlow",
)


def test_smoothness():
    """Test for smoothness operator"""
    tensor = tl.tensor([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.2, -3.4]])
    copy_tensor = tl.copy(tensor)
    parameter = 0.2
    tol = 0.5
    res = smoothness_prox(copy_tensor, parameter)
    true_res = tl.tensor([[1.2, 0.83, 0.98], [3.11, -4.12, 0.57], [0.58, 0.26, -2.51]])
    error = tl.norm(true_res - res, 2) / tl.norm(true_res, 2)
    assert_(error < tol)
    # Check that we did not change the original tensor
    assert_array_equal(copy_tensor, tensor)


def test_hard_thresholding():
    """Test for hard_thresholding operator"""
    tensor = tl.tensor([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.2, -3.4]])
    copy_tensor = tl.copy(tensor)
    threshold = 2
    res = hard_thresholding(tensor, threshold)
    true_res = tl.tensor([[0, 0, 0], [4.0, -6.0, 0], [0, 0, 0]])
    assert_array_almost_equal(true_res, res)
    # Check that we did not change the original tensor
    assert_array_equal(copy_tensor, tensor)


def test_soft_sparsity():
    """Test for soft_sparsity operator"""
    tensor = tl.tensor([[0.5, 1.3, 4.5], [0.8, 0.3, 2]])
    threshold = 2
    res = soft_sparsity_prox(tensor, threshold)
    true_res = tl.tensor([[0.85, 1.5, 2.0], [1.15, 0.5, 0.0]])
    assert_array_almost_equal(true_res, res)


def test_simplex():
    """Test for simplex operator"""
    # 1d Tensor
    tensor = tl.tensor([0.4, 0.5, 0.6])
    res = simplex_prox(tensor, 1)
    true_res = tl.tensor([0.23, 0.33, 0.43])
    assert_array_almost_equal(true_res, res, decimal=2)

    # 2d Tensor
    tensor = tl.tensor([[0.4, 1.5, 1], [0.5, 2, 3], [0.6, 0.3, 2.9]])
    res = simplex_prox(tensor, 1)
    true_res = tl.tensor([[0.23, 0.25, 0], [0.33, 0.75, 0.55], [0.43, 0, 0.45]])
    assert_array_almost_equal(true_res, res, decimal=2)


def test_normalized_sparsity():
    """Test for normalized_sparsity operator"""
    tensor = tl.tensor([2.0, 3.0, 4.0])
    res = normalized_sparsity_prox(tensor, 2)
    true_res = tl.tensor([0, 0.6, 0.8])
    assert_array_almost_equal(true_res, res)


def test_monotonicity():
    """Test for monotonicity operator"""
    tensor = tl.tensor(np.random.rand(20, 10))
    # Monotone increasing
    tensor_monoton = monotonicity_prox(tensor)
    assert_(np.all(np.diff(tensor_monoton, axis=0) >= 0))
    # Monotone decreasing
    tensor_monoton = monotonicity_prox(tensor, decreasing=True)
    assert_(np.all(np.diff(tensor_monoton, axis=0) <= 0))


def test_unimodality():
    """Test for unimdality operator"""
    tensor = tl.tensor(np.random.rand(10, 10))
    tensor_unimodal = unimodality_prox(tensor)
    for i in range(10):
        max_location = tl.argmax(tensor_unimodal[:, i])
        if max_location == 0:
            assert_(np.all(np.diff(tensor_unimodal[:, i], axis=0) <= 0))
        elif max_location == (tl.shape(tensor)[0] - 1):
            assert_(np.all(np.diff(tensor_unimodal[:, i], axis=0) >= 0))
        else:
            assert_(
                np.all(np.diff(tensor_unimodal[: int(max_location), i], axis=0) >= 0)
            )
            assert_(
                np.all(np.diff(tensor_unimodal[int(max_location) :, i], axis=0) <= 0)
            )


def test_l2_prox():
    """Test for l2 prox operator"""
    tensor = tl.tensor([2.0, 4.0, 4.0])
    res = l2_prox(tensor, 3)
    true_res = tl.tensor([1.0, 2.0, 2.0])
    assert_array_almost_equal(true_res, res)


def test_squared_l2_prox():
    """Test for squared l2 prox operator"""
    tensor = tl.tensor([3.0, 6.0, 9.0])
    res = l2_square_prox(tensor, 0.5)
    true_res = tl.tensor([1.5, 3.0, 4.5])
    assert_array_almost_equal(true_res, res)


def test_soft_thresholding():
    """Test for shrinkage"""
    # small test
    tensor = tl.tensor(
        [
            [[1, 2, 3], [4.3, -1.2, 3]],
            [[0.5, -5, -1.3], [1.2, 3.7, -9]],
            [[-2, 0, 1.0], [0.5, -0.5, 1.1]],
        ]
    )
    threshold = 1.1
    copy_tensor = tl.copy(tensor)
    res = soft_thresholding(tensor, threshold)
    true_res = tl.tensor(
        [
            [[0, 0.9, 1.9], [3.2, -0.1, 1.9]],
            [[0, -3.9, -0.2], [0.1, 2.6, -7.9]],
            [[-0.9, 0, 0], [0, -0, 0]],
        ]
    )
    # account for floating point errors: np array have a precision of around 2e-15
    # check np.finfo(np.float64).eps
    assert_array_almost_equal(true_res, res)
    # Check that we did not change the original tensor
    assert_array_equal(copy_tensor, tensor)

    # Another test
    tensor = tl.tensor([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.02, -3.4]])
    copy_tensor = tl.copy(tensor)
    threshold = 1.1
    true_res = tl.tensor([[0, 0.9, 0.4], [2.9, -4.9, 0], [0, 0, -2.3]])
    res = soft_thresholding(tensor, threshold)
    assert_array_almost_equal(true_res, res)
    assert_array_equal(copy_tensor, tensor)

    # Test with missing values
    tensor = tl.tensor([[1, 2, 1.5], [4, -6, -0.5], [0.2, 1.02, -3.4]])
    copy_tensor = tl.copy(tensor)
    mask = tl.tensor([[0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0]])
    threshold = 1.1 * mask
    true_res = tl.tensor([[1, 0.9, 0.4], [2.9, -6, 0], [0, 0, -3.4]])
    res = soft_thresholding(tensor, threshold)
    assert_array_almost_equal(true_res, res)
    assert_array_equal(copy_tensor, tensor)


def test_svd_thresholding():
    """Test for singular_value_thresholding operator"""
    U = tl.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    singular_values = tl.tensor([0.4, 2.1, -2.0])
    tensor = tl.dot(U, tl.reshape(singular_values, (-1, 1)) * tl.transpose(U))
    shrinked_singular_values = tl.tensor([0, 1.6, -1.5])
    true_res = tl.dot(
        U, tl.reshape(shrinked_singular_values, (-1, 1)) * tl.transpose(U)
    )
    res = svd_thresholding(tensor, 0.5)
    assert_array_almost_equal(true_res, res)


def test_procrustes():
    """Test for procrustes operator"""
    U = tl.tensor(np.random.rand(20, 10))
    S, _, V = truncated_svd(U, n_eigenvecs=min(U.shape))
    true_res = tl.dot(S, V)
    res = procrustes(U)
    assert_array_almost_equal(true_res, res)
