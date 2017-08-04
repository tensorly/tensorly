from ... import backend as T
from ..regression import MSE, RMSE

def test_MSE():
    """Test for MSE"""
    y_true = T.tensor([1, 0, 1.5, 0.5])
    y_pred = T.tensor([1, 1, 1, 1])
    true_mse = 1.5
    T.assert_(MSE(y_true, y_pred), true_mse)


def test_RMSE():
    """Test for RMSE"""
    y_true = T.tensor([1, 2, 0.5, 1.5, 0.5, -1, -1])
    y_pred = T.tensor([1, 2, 1, 1, 1, -0.5, 0])
    true_mse = 2
    T.assert_(MSE(y_true, y_pred), true_mse)
