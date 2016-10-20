import numpy as np
from numpy.testing import assert_
from ..tucker_regression import TuckerRegressor
from ...base import tensor_to_vec, partial_tensor_to_vec
from ...metrics.regression import RMSE


def test_TuckerRegressor():
    """Test for TuckerRegressor"""

    # Parameter of the experiment
    image_height = 10
    image_width = 10
    n_channels = 3
    ranks = [5, 5, 2]
    tol = 10e-3

    # Generate random samples
    X = np.random.normal(size=(1200, image_height, image_width, n_channels), loc=0, scale=1)
    regression_weights = np.zeros((image_height, image_width, n_channels))
    regression_weights[2:-2, 2:-2, 0] = 1
    regression_weights[2:-2, 2:-2, 1] = 2
    regression_weights[2:-2, 2:-2, 2] = -1

    y = partial_tensor_to_vec(X, skip_begin=1).dot(tensor_to_vec(regression_weights))
    X_train = X[:1000, ...]
    X_test = X[1000:, ...]
    y_train = y[:1000]
    y_test = y[1000:]

    estimator = TuckerRegressor(weight_ranks=ranks, tol=10e-8, reg_W=1, n_iter_max=200, verbose=True)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    error = RMSE(y_test, y_pred)
    assert_(error <= tol, msg='Tucker Regression : RMSE={} > {}'.format(error, tol))

    params = estimator.get_params()
    assert_(params['weight_ranks'] == [5, 5, 2], msg='get_params did not return the correct parameters')
    params['weight_ranks'] = [5, 5, 5]
    estimator.set_params(**params)
    assert_(estimator.weight_ranks == [5, 5, 5], msg='set_params did not correctly set the given parameters')