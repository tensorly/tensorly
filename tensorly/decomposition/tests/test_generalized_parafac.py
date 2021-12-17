from .._generalized_parafac import generalized_parafac, stochastic_generalized_parafac, Stochastic_GCP, GCP
from ...metrics.losses import loss_operator
from ...testing import assert_, assert_class_wrapper_correctly_passes_arguments
from ... import backend as T
from ...cp_tensor import cp_to_tensor
from ...random import random_cp
import pytest
import tensorly as tl


def test_generalized_parafac(monkeypatch):
    """Test for the Generalized Parafac decomposition
    """
    tol_norm_2 = 0.3
    rank = 3
    shape = [8, 10, 6]
    init = 'random'
    rng = T.check_random_state(1234)
    initial_tensor = cp_to_tensor(random_cp(shape, rank=rank))

    # Gaussian
    loss = 'gaussian'
    gcp_result = generalized_parafac(initial_tensor, loss=loss, rank=rank, init=init, tol=1e-5)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Gamma
    loss = 'gamma'
    array = rng.gamma(1, initial_tensor, size=shape)
    tensor = T.tensor(array)
    gcp_result = generalized_parafac(tensor, loss=loss, rank=rank, init=init, tol=1e-8)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Rayleigh
    loss = 'rayleigh'
    array = rng.rayleigh(initial_tensor, size=shape)
    tensor = T.tensor(array)
    gcp_result = generalized_parafac(tensor, loss=loss, rank=rank, init=init, tol=1e-16, lr=1e-5)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-count
    loss = 'poisson_count'
    array = 1.0 * rng.poisson(initial_tensor, size=shape)
    tensor = T.tensor(array)
    gcp_result = generalized_parafac(tensor, loss=loss, rank=rank, init=init, tol=1e-8)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-log
    loss = 'poisson_log'
    array = 1.0 * rng.poisson(tl.exp(initial_tensor), size=shape)
    tensor = T.tensor(array)
    gcp_result = generalized_parafac(tensor, loss=loss, rank=rank, init=init, tol=1e-5)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-odds
    loss = 'bernoulli_odds'
    array = 1.0 * rng.binomial(1, initial_tensor / (initial_tensor + 1), size=shape)
    tensor = T.tensor(array)
    gcp_result = generalized_parafac(tensor, loss=loss, rank=rank, init=init, tol=1e-5)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-logit
    loss = 'bernoulli_logit'
    array = 1.0 * rng.binomial(1, tl.exp(initial_tensor) / (tl.exp(initial_tensor) + 1), size=shape)
    tensor = T.tensor(array)
    gcp_result = generalized_parafac(tensor, loss=loss, rank=rank, init=init, tol=1e-5)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, generalized_parafac, GCP, rank=3)


@pytest.mark.xfail(tl.get_backend() == 'tensorflow', reason='Fails on tensorflow')
def test_stochastic_generalized_parafac(monkeypatch):
    """Test for the Stochastic Generalized Parafac decomposition
    """
    tol_norm_2 = 0.3
    rank = 3
    shape = [8, 10, 6]
    init = 'random'
    rng = T.check_random_state(1234)
    initial_tensor = cp_to_tensor(random_cp(shape, rank=rank))
    batch_size = 8

    # Gaussian
    loss = 'gaussian'
    gcp_result = stochastic_generalized_parafac(initial_tensor, loss=loss, rank=rank, n_iter_max=100, init=init)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Gamma
    loss = 'gamma'
    array = rng.gamma(1, initial_tensor, size=shape)
    tensor = T.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=100, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Rayleigh
    loss = 'rayleigh'
    array = rng.rayleigh(initial_tensor, size=shape)
    tensor = T.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=100, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-count
    loss = 'poisson_count'
    array = 1.0 * rng.poisson(initial_tensor, size=shape)
    tensor = T.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=500, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Poisson-log
    loss = 'poisson_log'
    array = 1.0 * rng.poisson(tl.exp(initial_tensor), size=shape)
    tensor = T.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=500, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-odds
    loss = 'bernoulli_odds'
    array = 1.0 * rng.binomial(1, initial_tensor / (initial_tensor + 1), size=shape)
    tensor = T.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, n_iter_max=100, batch_size=batch_size)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')

    # Bernoulli-logit
    loss = 'bernoulli_logit'
    array = 1.0 * rng.binomial(1, tl.exp(initial_tensor) / (tl.exp(initial_tensor) + 1), size=shape)
    tensor = T.tensor(array)
    gcp_result = stochastic_generalized_parafac(tensor, loss=loss, rank=rank, init=init, batch_size=batch_size,
                                                epochs=100, n_iter_max=100)
    reconstructed_tensor = cp_to_tensor(gcp_result)
    error = loss_operator(initial_tensor, reconstructed_tensor, loss)
    error = T.sum(error) / T.norm(initial_tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, stochastic_generalized_parafac, Stochastic_GCP, rank=3)
