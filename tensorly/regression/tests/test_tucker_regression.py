import numpy as np
import pytest

from ..tucker_regression import TuckerRegressor
from ...base import tensor_to_vec, partial_tensor_to_vec
from ...metrics.regression import RMSE
from ... import backend as T
from ...testing import assert_, assert_allclose


@pytest.mark.parametrize("rank,regularization", [(1, 0.0), (1, 0.2), (2, 0.2)])
def test_single_mode_tucker_regression(rank, regularization):
    """A feature matrix agrees with independent dense alternating ridge fits."""
    rng = np.random.RandomState(19)
    values = rng.normal(size=(32, 5))
    targets = values @ rng.normal(size=5)
    seed = 31
    core = np.random.RandomState(seed).normal(size=rank)
    for _ in range(4):
        design = np.kron(values, core.reshape(1, -1))
        factors = np.linalg.solve(
            design.T @ design + regularization * np.eye(5 * rank),
            design.T @ targets,
        ).reshape(5, rank)
        design = values @ factors
        core = np.linalg.solve(
            design.T @ design + regularization * np.eye(rank),
            design.T @ targets,
        )
    expected = factors @ core
    X, y = T.tensor(values), T.tensor(targets)
    before_X, before_y = T.copy(X), T.copy(y)
    estimator = TuckerRegressor(
        [rank],
        reg_W=regularization,
        n_iter_max=4,
        tol=0,
        random_state=seed,
        verbose=False,
    ).fit(X, y)
    assert_allclose(estimator.weight_tensor_, expected, rtol=1e-5, atol=1e-6)
    assert_allclose(estimator.predict(X), values @ expected, rtol=1e-5, atol=1e-6)
    assert_allclose(X, before_X, rtol=0, atol=0)
    assert_allclose(y, before_y, rtol=0, atol=0)


def test_TuckerRegressor():
    """Test for TuckerRegressor"""

    # Parameter of the experiment
    image_height = 10
    image_width = 10
    n_channels = 3
    ranks = [5, 5, 2]
    tol = 0.05

    # Generate random samples
    X = T.tensor(
        np.random.normal(
            size=(1200, image_height, image_width, n_channels), loc=0, scale=1
        )
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

    estimator = TuckerRegressor(
        weight_ranks=ranks, tol=10e-8, reg_W=1, n_iter_max=200, verbose=True
    )
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    error = RMSE(y_test, y_pred)
    assert_(error <= tol, msg=f"Tucker Regression : RMSE={error} > {tol}")

    params = estimator.get_params()
    assert_(
        params["weight_ranks"] == [5, 5, 2],
        msg="get_params did not return the correct parameters",
    )
    params["weight_ranks"] = [5, 5, 5]
    estimator.set_params(**params)
    assert_(
        estimator.weight_ranks == [5, 5, 5],
        msg="set_params did not correctly set the given parameters",
    )
