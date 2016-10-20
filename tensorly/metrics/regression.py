import numpy as np

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


def MSE(y_true, y_pred):
    """Returns the mean squared error between the two predictions

    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Returns
    -------
    float
    """
    return np.mean((y_true - y_pred) ** 2)


def RMSE(y_true, y_pred):
    """Returns the regularised mean squared error between the two predictions
    (the square-root is applied to the mean_squared_error)

    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Returns
    -------
    float
    """
    return np.sqrt(MSE(y_true, y_pred))

