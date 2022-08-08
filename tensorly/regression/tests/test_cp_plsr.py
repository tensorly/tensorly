import numpy as np
import pytest
import tensorly as tl

from ... import backend as T
from ...testing import assert_allclose
from ...base import partial_tensor_to_vec, tensor_to_vec
from ...cp_tensor import CPTensor, cp_normalize, cp_to_tensor
from ...metrics.factors import congruence_coefficient
from ...metrics.regression import RMSE
from ...random import random_cp
from ..cp_plsr import CP_PLSR

skip_if_backend = pytest.mark.skipif(
    tl.get_backend() in ("tensorflow"),
    reason=f"Operation not supported in {tl.get_backend()}",
)

# Authors: Jackson L. Chin, Cyrillus Tan, Aaron Meyer


TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 8
N_LATENT = 8

TEST_RANKS = [3, 4, 5, 6]
TEST_RESPONSE = [1, 4, 8, 16, 32]

RANDOM_STATE = np.random.RandomState(215)


# Supporting Functions


def _get_pls_dataset(tensor_dimensions, n_latent, n_response):
    x_tensor = random_cp(
        tensor_dimensions,
        n_latent,
        orthogonal=True,
        normalise_factors=True,
        random_state=RANDOM_STATE,
    )
    y_tensor = random_cp(
        (tensor_dimensions[0], n_response), n_latent, random_state=RANDOM_STATE
    )

    y_tensor.factors[0] = x_tensor.factors[0]
    x = cp_to_tensor(x_tensor)
    y = cp_to_tensor(y_tensor)

    return x, y, x_tensor, y_tensor


def _get_standard_synthetic():
    return _get_pls_dataset(TENSOR_DIMENSIONS, N_LATENT, N_RESPONSE)


# Class Structure Tests


@pytest.mark.parametrize("x_rank", TEST_RANKS)
@pytest.mark.parametrize("n_response", TEST_RESPONSE)
def test_transform(x_rank, n_response):
    x, y, _, _ = _get_pls_dataset(tuple([10] * x_rank), N_LATENT, n_response)
    pls = CP_PLSR(N_LATENT)
    pls.fit(x, y)

    transformed = pls.transform(x)
    assert_allclose(transformed, pls.X_factors[0])


def test_factor_normality():
    x, y, _, _ = _get_standard_synthetic()
    pls = CP_PLSR(n_components=N_LATENT)
    pls.fit(x, y)

    for x_factor in pls.X_factors[1:]:
        assert_allclose(tl.norm(x_factor, axis=0), 1, rtol=2e-7)

    for y_factor in pls.Y_factors[1:]:
        assert_allclose(tl.norm(y_factor, axis=0), 1)


def test_factor_orthogonality():
    x, y, _, _ = _get_standard_synthetic()
    pls = CP_PLSR(n_components=N_LATENT)
    pls.fit(x, y)
    x_cp = CPTensor((None, pls.X_factors))
    x_cp = cp_normalize(x_cp)

    for component_1 in range(x_cp.rank):
        for component_2 in range(component_1 + 1, x_cp.rank):
            factor_product = 1
            for factor in x_cp.factors:
                factor_product *= np.dot(factor[:, component_1], factor[:, component_2])
            assert abs(factor_product) < 1e-8


def test_consistent_components():
    x, y, _, _ = _get_standard_synthetic()
    pls = CP_PLSR(n_components=N_LATENT)
    pls.fit(x, y)

    for x_factor in pls.X_factors:
        assert x_factor.shape[1] == N_LATENT

    for y_factor in pls.Y_factors:
        assert y_factor.shape[1] == N_LATENT


# Dimension Compatibility Tests


@pytest.mark.parametrize("x_rank", TEST_RANKS)
@pytest.mark.parametrize("n_response", TEST_RESPONSE)
def test_dimension_compatibility(x_rank, n_response):
    x, y, _, _ = _get_pls_dataset(tuple([10] * x_rank), N_LATENT, n_response)
    try:
        pls = CP_PLSR(N_LATENT)
        pls.fit(x, y)
    except ValueError:
        raise AssertionError(
            f"Fit failed for {len(x.shape)}-dimensional tensor with "
            f"{n_response} response variables in y"
        )


# Decomposition Accuracy Tests


@skip_if_backend
def test_zero_covariance_x():
    x, y, _, _ = _get_standard_synthetic()
    x = tl.index_update(x, tl.index[:, 0, :], 1)
    pls = CP_PLSR(N_LATENT)
    pls.fit(x, y)

    assert_allclose(pls.X_factors[1][0, :], 0, atol=1e-6)


@skip_if_backend
def test_zero_covariance_y():
    x, y, _, _ = _get_standard_synthetic()
    y = tl.index_update(y, tl.index[:, 0], 1)
    pls = CP_PLSR(N_LATENT)
    pls.fit(x, y)

    assert_allclose(pls.Y_factors[1][0, :], 0)


@pytest.mark.parametrize("x_rank", TEST_RANKS)
@pytest.mark.parametrize("n_response", TEST_RESPONSE)
def test_decomposition_accuracy(x_rank, n_response):
    x, y, x_cp, y_cp = _get_pls_dataset(tuple([10] * x_rank), N_LATENT, n_response)
    pls = CP_PLSR(N_LATENT)
    pls.fit(x, y)

    cp_normalize(x_cp)

    for pls_factor, true_factor in zip(pls.X_factors, x_cp.factors):
        assert congruence_coefficient(pls_factor, true_factor)[0] > 0.85

    assert congruence_coefficient(pls.Y_factors[1], y_cp.factors[1])[0] > 0.85


def test_reconstruction_x():
    x, y, _, _ = _get_standard_synthetic()
    pls = CP_PLSR(N_LATENT)
    pls.fit(x, y)

    x_cp = CPTensor((None, pls.X_factors))
    reconstructed_x = x_cp.to_tensor()

    assert_allclose(reconstructed_x, x, rtol=0, atol=1e-2)


@skip_if_backend
@pytest.mark.parametrize("vars_shape", [(8, 8, 3), (8, 8, 9, 3)])
def test_CPRegressor(vars_shape):
    """Test for CP_PLSR."""
    # Generate random samples
    rng = T.check_random_state(1234)
    X = random_cp((400, *vars_shape), rank=2, random_state=rng, full=True)
    regression_weights = T.zeros(vars_shape, **T.context(X))
    regression_weights = T.index_update(regression_weights, T.index[2:-2, 2:-2, 0], 1)
    regression_weights = T.index_update(regression_weights, T.index[2:-2, 2:-2, 1], 2)
    regression_weights = T.index_update(regression_weights, T.index[2:-2, 2:-2, 2], -1)

    y = T.dot(partial_tensor_to_vec(X, skip_begin=1), tensor_to_vec(regression_weights))
    X_train = X[:200, :, :]
    X_test = X[200:, :, :]
    y_train = y[:200]
    y_test = y[200:]

    estimator = CP_PLSR(n_components=4, verbose=True)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    estimator.transform(X_train, y_train)
    error = RMSE(y_test, y_pred)
    print(error)
    # TODO: Add assertion about expected error.
