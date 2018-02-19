import numpy as np
from ... import backend as T
from ..regression import MSE, RMSE, correlation

def test_MSE():
    """Test for MSE"""
    y_true = T.tensor([1, 0, 2, -2])
    y_pred = T.tensor([1, -1, 1, -1])
    true_mse = 0.75
    T.assert_array_almost_equal(MSE(y_true, y_pred), true_mse)

def test_RMSE():
    """Test for RMSE"""
    y_true = T.tensor([1, 0, 2, -2])
    y_pred = T.tensor([0, -1, 1, -1])
    true_mse = 1
    T.assert_array_almost_equal(MSE(y_true, y_pred), true_mse)

def test_correlation():
    """Test for correlation"""
    a = T.tensor(np.random.random(10))
    b = T.tensor(np.random.random(10))
    T.assert_array_almost_equal(correlation(a, a*2+1), 1)
    T.assert_array_almost_equal(correlation(a, -a*2+1), -1)
    
    a = T.tensor([1, 2, 3, 2, 1])
    b = T.tensor([1, 2, 3, 4, 5])
    T.assert_array_almost_equal(correlation(a, b), 0)

    a = T.tensor([[1, 2, 3, 2, 1]]*3)
    b = T.tensor([[1, 2, 3, 4, 5]]*3)
    res = T.tensor([0, 0, 0])
    T.assert_array_almost_equal(correlation(a, b, axis=1), res)


