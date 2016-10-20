import numpy as np
from numpy.testing import assert_
from ..regression import MSE, RMSE

def test_MSE():
    """Test for MSE"""
    y_true = np.array([1, 0, 1.5, 0.5])
    y_pred = np.array([1, 1, 1, 1])
    true_mse = 1.5
    assert_(MSE(y_true, y_pred), true_mse)


def test_RMSE():
    """Test for RMSE"""
    y_true = np.array([1, 2, 0.5, 1.5, 0.5, -1, -1])
    y_pred = np.array([1, 2, 1, 1, 1, -0.5, 0])
    true_mse = 2
    assert_(MSE(y_true, y_pred), true_mse)