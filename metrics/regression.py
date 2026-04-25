from .. import backend as T

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


def MSE(y_true, y_pred, axis=None):
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
    return T.mean((y_true - y_pred) ** 2, axis=axis)


def RMSE(y_true, y_pred, axis=None):
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
    return T.sqrt(MSE(y_true, y_pred, axis=axis))


def R2_score(X_original, X_predicted):
    """Returns the R^2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because prediction can be
    arbitrarily worse).

    Parameters
    ----------
    X_original: array
        The original array
    X_predicted: array
        Thre predicted array.

    Returns
    -------
    float
    """
    return 1 - T.norm(X_predicted - X_original) ** 2.0 / T.norm(X_original) ** 2.0


def reflective_correlation_coefficient(y_true, y_pred, axis=None):
    """Reflective variant of Pearson's product moment correlation coefficient
    where the predictions are not centered around their mean values.

    Parameters
    ----------
    y_true : array of shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array of shape (n_samples, )
        Estimated target values.

    Returns
    -------
    float: reflective correlation coefficient
    """
    return T.sum(y_true * y_pred, axis=axis) / T.sqrt(
        T.sum(y_true**2, axis=axis) * T.sum(y_pred**2, axis=axis)
    )


def covariance(y_true, y_pred, axis=None):
    centered_true = T.mean(y_true, axis=axis)
    centered_pred = T.mean(y_pred, axis=axis)

    if axis is not None:
        # TODO: write a function to do this..
        shape = list(T.shape(y_true))
        shape[axis] = 1
        centered_true = T.reshape(centered_true, shape)
        shape = list(T.shape(y_pred))
        shape[axis] = 1
        centered_pred = T.reshape(centered_pred, shape)

    return T.mean((y_true - centered_true) * (y_pred - centered_pred), axis=axis)


def variance(y, axis=None):
    return covariance(y, y, axis=axis)


def standard_deviation(y, axis=None):
    return T.sqrt(variance(y, axis=axis))


def correlation(y_true, y_pred, axis=None):
    """Pearson's product moment correlation coefficient"""
    return covariance(y_true, y_pred, axis=axis) / T.sqrt(
        variance(y_true, axis) * variance(y_pred, axis)
    )
