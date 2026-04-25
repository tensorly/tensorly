import numpy as np

import tensorly as tl
from ..regression import MSE, RMSE, R2_score, correlation
from ...testing import assert_array_almost_equal


def test_MSE():
    """Test for MSE"""
    y_true = tl.tensor([1, 0, 2, -2], dtype=tl.float32)
    y_pred = tl.tensor([1, -1, 1, -1], dtype=tl.float32)
    true_mse = 0.75
    assert_array_almost_equal(MSE(y_true, y_pred), true_mse)


def test_RMSE():
    """Test for RMSE"""
    y_true = tl.tensor([1, 0, 2, -2], dtype=tl.float32)
    y_pred = tl.tensor([0, -1, 1, -1], dtype=tl.float32)
    true_mse = 1
    assert_array_almost_equal(RMSE(y_true, y_pred), true_mse)


def test_R2_score():
    """Test for RMSE"""
    X_original = tl.randn((5, 4, 3))
    assert R2_score(X_original, X_original) == 1.0
    assert R2_score(X_original, X_original * 2) == 0.0
    assert R2_score(X_original, tl.zeros_like(X_original)) == 0.0


def test_correlation():
    """Test for correlation"""
    a = tl.tensor(np.random.random(10))
    b = tl.tensor(np.random.random(10))
    assert_array_almost_equal(correlation(a, a * 2 + 1), 1)
    assert_array_almost_equal(correlation(a, -a * 2 + 1), -1)

    a = tl.tensor([1, 2, 3, 2, 1], dtype=tl.float32)
    b = tl.tensor([1, 2, 3, 4, 5], dtype=tl.float32)
    assert_array_almost_equal(correlation(a, b), 0)

    a = tl.tensor([[1, 2, 3, 2, 1]] * 3, dtype=tl.float32)
    b = tl.tensor([[1, 2, 3, 4, 5]] * 3, dtype=tl.float32)
    res = tl.tensor([0, 0, 0], dtype=tl.float32)
    assert_array_almost_equal(correlation(a, b, axis=1), res)
