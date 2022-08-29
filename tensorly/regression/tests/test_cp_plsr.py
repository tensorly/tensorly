import numpy as np
import pytest
import tensorly as tl

from ...testing import assert_allclose
from ...cp_tensor import CPTensor, cp_normalize, cp_to_tensor
from ...metrics.factors import congruence_coefficient
from ...random import random_cp
from ..cp_plsr import CP_PLSR


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
        dtype=tl.float64,
    )
    y_tensor = random_cp(
        (tensor_dimensions[0], n_response),
        n_latent,
        random_state=RANDOM_STATE,
        dtype=tl.float64,
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
        assert_allclose(tl.norm(x_factor, axis=0), 1, rtol=1e-6)

    for y_factor in pls.Y_factors[1:]:
        assert_allclose(tl.norm(y_factor, axis=0), 1, rtol=1e-6)


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


def test_zero_covariance_x():
    x, y, _, _ = _get_standard_synthetic()
    x = np.copy(tl.to_numpy(x))  # workaround to make this work for all backends
    x[:, 0, :] = 1
    x = tl.tensor(x, **tl.context(y))
    pls = CP_PLSR(N_LATENT)
    pls.fit(x, y)

    assert_allclose(pls.X_factors[1][0, :], 0, atol=1e-6)


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


def test_optimized_rand_covariance():
    x = tl.tensor(np.random.rand(80, 60, 50, 40))
    y = tl.tensor(np.random.rand(80))

    pls = CP_PLSR(3)
    pls.fit(x, y)
    y = tl.tensor_to_vec(y)

    for component in range(3):
        assert np.corrcoef(pls.X_factors[0][:, component], y)[0, 1] > 0.0


@pytest.mark.parametrize("n_latent", np.arange(1, 11))
def test_optimized_covariance(n_latent):
    x, y, x_cp, _ = _get_pls_dataset(TENSOR_DIMENSIONS, n_latent, 1)
    pls = CP_PLSR(n_latent)
    pls.fit(x, y)
    y = tl.tensor_to_vec(y)

    max_cov = 0
    pls_cov = 0
    for component in np.arange(n_latent):
        max_cov += abs(
            np.cov(tl.tensor_to_vec(x_cp.factors[0][:, component]), y, bias=True)[0, 1]
        )
        pls_cov += abs(
            np.cov(tl.tensor_to_vec(pls.X_factors[0][:, component]), y, bias=True)[0, 1]
        )

    assert_allclose(max_cov, pls_cov, rtol=2e-6, atol=2e-6)
