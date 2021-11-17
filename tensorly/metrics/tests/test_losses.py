import tensorly as tl
from ...testing import assert_array_almost_equal
from ..losses import loss_operator, gradient_operator


def test_loss_operator():
    """Test for loss operator"""

    tensor_orig = tl.tensor([1, 0, 2, 2])
    tensor_est = tl.tensor([1, 1, 1, 1])

    # Gaussian loss
    true_loss = [0, 0.25, 0.25, 0.25]
    assert_array_almost_equal(loss_operator(tensor_orig, tensor_est, loss='gaussian'), true_loss)

    # Gamma loss
    true_loss = [0.25, 0, 0.5, 0.5]
    assert_array_almost_equal(loss_operator(tensor_orig, tensor_est, loss='gamma'), true_loss)

    # Rayleigh loss
    true_loss = [0.19, 0, 0.78, 0.78]
    assert_array_almost_equal(loss_operator(tensor_orig, tensor_est, loss='rayleigh'), true_loss, decimal=2)

    # Poisson-log loss
    true_loss = [0.42, 0.67, 0.17, 0.17]
    assert_array_almost_equal(loss_operator(tensor_orig, tensor_est, loss='poisson_log'), true_loss, decimal=2)

    # Poisson-count loss
    true_loss = [0.25, 0.25, 0.25, 0.25]
    assert_array_almost_equal(loss_operator(tensor_orig, tensor_est, loss='poisson_count'), true_loss)

    # Bernoulli-odds loss
    true_loss = [0.17, 0.17, 0.17, 0.17]
    assert_array_almost_equal(loss_operator(tensor_orig, tensor_est, loss='bernoulli_odds'), true_loss, decimal=2)

    # Bernoulli-logit loss
    true_loss = [0.07, 0.32, -0.17, -0.17]
    assert_array_almost_equal(loss_operator(tensor_orig, tensor_est, loss='bernoulli_logit'), true_loss, decimal=2)


def test_gradient_operator():
    """Test for gradient operator"""
    tensor_orig = tl.tensor([1, 0, 2, 2])
    tensor_est = tl.tensor([1, 1, 1, 1])

    # Gaussian gradient
    true_gradient = [0, 0.5, -0.5, -0.5]
    assert_array_almost_equal(gradient_operator(tensor_orig, tensor_est, loss='gaussian'), true_gradient)

    # Gamma gradient
    true_gradient = [0, 0.25, -0.25, -0.25]
    assert_array_almost_equal(gradient_operator(tensor_orig, tensor_est, loss='gamma'), true_gradient)

    # Rayleigh gradient
    true_gradient = [0.1, 0.5, -1.07, -1.07]
    assert_array_almost_equal(gradient_operator(tensor_orig, tensor_est, loss='rayleigh'), true_gradient, decimal=2)

    # Poisson-log gradient
    true_gradient = [0.42, 0.67, 0.17, 0.17]
    assert_array_almost_equal(gradient_operator(tensor_orig, tensor_est, loss='poisson_log'), true_gradient, decimal=2)

    # Poisson-count gradient
    true_gradient = [0, 0.25, -0.25, -0.25]
    assert_array_almost_equal(gradient_operator(tensor_orig, tensor_est, loss='poisson_count'), true_gradient)

    # Bernoulli-odds gradient
    true_gradient = [-0.125,  0.125, -0.375, -0.375]
    assert_array_almost_equal(gradient_operator(tensor_orig, tensor_est, loss='bernoulli_odds'), true_gradient, decimal=3)

    # Bernoulli-logit gradient
    true_gradient = [-0.06,  0.18, -0.31, -0.31]
    assert_array_almost_equal(gradient_operator(tensor_orig, tensor_est, loss='bernoulli_logit'), true_gradient, decimal=2)
