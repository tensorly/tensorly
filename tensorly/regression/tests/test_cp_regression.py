import numpy as np

from ..cp_regression import CPRegressor
from ...base import tensor_to_vec, partial_tensor_to_vec
from ...metrics.regression import RMSE
from ... import backend as T
from ...random import random_cp
from ...testing import assert_, assert_allclose


def test_CPRegressor():
    """Test for CPRegressor"""

    # Parameter of the experiment
    image_height = 8
    image_width = 8
    n_channels = 3
    tol = 0.05

    # Generate random samples
    rng = T.check_random_state(1234)
    X = T.tensor(
        rng.normal(size=(1200, image_height, image_width, n_channels), loc=0, scale=1)
    )
    regression_weights = np.zeros((image_height, image_width, n_channels))
    regression_weights[2:-2, 2:-2, 0] = 1
    regression_weights[2:-2, 2:-2, 1] = 2
    regression_weights[2:-2, 2:-2, 2] = -1
    regression_weights = T.tensor(regression_weights)

    y = T.dot(partial_tensor_to_vec(X, skip_begin=1), tensor_to_vec(regression_weights))
    X_train = X[:1000, :, :]
    X_test = X[1000:, :, :]
    y_train = y[:1000]
    y_test = y[1000:]

    estimator = CPRegressor(
        weight_rank=4, tol=10e-8, reg_W=1, n_iter_max=200, verbose=True
    )
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    error = RMSE(y_test, y_pred)
    assert_(error <= tol, msg=f"CP Regressor : RMSE is too large, {error} > {tol}")

    params = estimator.get_params()
    assert_(
        params["weight_rank"] == 4,
        msg="get_params did not return the correct parameters",
    )
    params["weight_rank"] = 5
    estimator.set_params(**params)
    assert_(
        estimator.weight_rank == 5,
        msg="set_params did not correctly set the given parameters",
    )


def test_multidim_CPRegressor():
    tol = 1e-3
    rng = T.check_random_state(1234)

    regression_weights = random_cp(
        shape=(12, 5, 4, 4, 3), rank=3, full=True, random_state=rng
    )
    X = T.randn((1200, 12, 5, 4), seed=rng)
    y = T.reshape(
        T.dot(partial_tensor_to_vec(X), T.reshape(regression_weights, (-1, 4 * 3))),
        (-1, 4, 3),
    )
    X_train = X[:1000]
    X_test = X[1000:]
    y_train = y[:1000]
    y_test = y[1000:]

    estimator = CPRegressor(
        weight_rank=3, tol=1e-8, reg_W=0.0, n_iter_max=200, verbose=True
    )
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    error = RMSE(y_test, y_pred)
    assert_(error <= tol, msg=f"CP Regressor : RMSE is too large, {error} > {tol}")
    assert_allclose(
        estimator.weight_tensor_,
        regression_weights,
        atol=tol,
        err_msg="CPRegressor did not converge to the correct weights",
    )
