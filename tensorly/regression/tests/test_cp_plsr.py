import pytest

from ..cp_plsr import CP_PLSR
from ...base import tensor_to_vec, partial_tensor_to_vec
from ...metrics.regression import RMSE
from ...random import random_cp
from ... import backend as T
from ...testing import assert_


@pytest.mark.parametrize("vars_shape", [(8, 8, 3), (8, 8, 9, 3)])
def test_CPRegressor(vars_shape):
    """Test for CP_PLSR."""
    # Generate random samples
    rng = T.check_random_state(1234)
    X = random_cp((400, *vars_shape), rank=2, random_state=rng, full=True)
    regression_weights = T.zeros(vars_shape)
    regression_weights[2:-2, 2:-2, 0] = 1
    regression_weights[2:-2, 2:-2, 1] = 2
    regression_weights[2:-2, 2:-2, 2] = -1
    regression_weights = T.tensor(regression_weights)

    y = T.dot(partial_tensor_to_vec(X, skip_begin=1), tensor_to_vec(regression_weights))
    X_train = X[:200, :, :]
    X_test = X[200:, :, :]
    y_train = y[:200]
    y_test = y[200:]

    estimator = CP_PLSR(n_components=4, verbose=True)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    error = RMSE(y_test, y_pred)
    print(error)
    # TODO: Add assertion about expected error.
