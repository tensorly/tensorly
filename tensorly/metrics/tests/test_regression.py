import numpy as np

import tensorly as tl
from ..regression import MSE, RMSE, correlation
from ...testing import assert_array_almost_equal

def test_MSE():
    """Test for MSE"""
    y_true = tl.tensor([1, 0, 2, -2])
    y_pred = tl.tensor([1, -1, 1, -1])
    true_mse = 0.75
    assert_array_almost_equal(MSE(y_true, y_pred), true_mse)


def test_RMSE():
    """Test for RMSE"""
    y_true = tl.tensor([1, 0, 2, -2])
    y_pred = tl.tensor([0, -1, 1, -1])
    true_mse = 1
    assert_array_almost_equal(RMSE(y_true, y_pred), true_mse)


def test_correlation():
    """Test for correlation"""
    a = tl.tensor(np.random.random(10))
    b = tl.tensor(np.random.random(10))
    assert_array_almost_equal(correlation(a, a*2+1), 1)
    assert_array_almost_equal(correlation(a, -a*2+1), -1)

    a = tl.tensor([1, 2, 3, 2, 1])
    b = tl.tensor([1, 2, 3, 4, 5])
    assert_array_almost_equal(correlation(a, b), 0)

    a = tl.tensor([[1, 2, 3, 2, 1]]*3)
    b = tl.tensor([[1, 2, 3, 4, 5]]*3)
    res = tl.tensor([0, 0, 0])
    assert_array_almost_equal(correlation(a, b, axis=1), res)
