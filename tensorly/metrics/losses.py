import tensorly as tl
import math


def loss_operator(tensor, estimated_tensor, loss, mask=None):
    """
    Operator to use loss functions from [1] in order to compute loss for
    generalized parafac decomposition.

    Parameters
    ----------
    tensor : ndarray
    estimated_tensor : ndarray
    loss : {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.

    Returns
    -------
    error : ndarray
         Size based normalized loss for each entry
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    """
    if mask is not None:
        estimated_tensor = estimated_tensor * mask

    epsilon = 1e-8
    if loss == 'gaussian':
        error = (tensor - estimated_tensor) ** 2
    elif loss == 'bernoulli_odds':
        error = tl.log(estimated_tensor + 1) - (tensor * tl.log(estimated_tensor + epsilon))
    elif loss == 'bernoulli_logit':
        error = tl.log(tl.exp(estimated_tensor) + 1) - (tensor * estimated_tensor)
    elif loss == 'rayleigh':
        error = 2 * tl.log(estimated_tensor + epsilon) + (math.pi / 4) * ((tensor / (estimated_tensor + epsilon)) ** 2)
    elif loss == 'poisson_count':
        error = estimated_tensor - tensor * tl.log(estimated_tensor + epsilon)
    elif loss == 'poisson_log':
        error = tl.exp(estimated_tensor) - (tensor * estimated_tensor)
    elif loss == 'gamma':
        error = tensor / (estimated_tensor + epsilon) + tl.log(estimated_tensor + epsilon)
    else:
        raise ValueError('Loss "{}" not recognized'.format(loss))
    size = tl.tensor(tl.shape(tl.tensor_to_vec(tensor)), **tl.context(tensor))
    return error / size


def gradient_operator(tensor, estimated_tensor, loss, mask=None):
    """
    Operator to use loss functions from [1] in order to compute gradient for
    generalized parafac decomposition.

    Parameters
    ----------
    tensor : ndarray
    estimated_tensor : ndarray
    loss : {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.

    Returns
    -------
    gradient : ndarray
        Size based normalized gradient for each entry
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    """
    if mask is not None:
        estimated_tensor = estimated_tensor * mask
        tensor = tensor * mask

    epsilon = 1e-8
    if loss == 'gaussian':
        gradient = 2 * (estimated_tensor - tensor)
    elif loss == 'bernoulli_odds':
        gradient = 1 / (estimated_tensor + 1) - (tensor / (estimated_tensor + epsilon))
    elif loss == 'bernoulli_logit':
        gradient = tl.exp(estimated_tensor) / (tl.exp(estimated_tensor) + 1) - tensor
    elif loss == 'rayleigh':
        gradient = 2 / (estimated_tensor + epsilon) - (math.pi / 2) * (tensor ** 2) / ((estimated_tensor + epsilon) ** 3)
    elif loss == 'poisson_count':
        gradient = 1 - tensor / (estimated_tensor + epsilon)
    elif loss == 'poisson_log':
        gradient = tl.exp(estimated_tensor) - tensor
    elif loss == 'gamma':
        gradient = -tensor / ((estimated_tensor + epsilon) ** 2) + (1 / (estimated_tensor + epsilon))
    else:
        raise ValueError('Loss "{}" not recognized'.format(loss))
    size = tl.tensor(tl.shape(tl.tensor_to_vec(tensor)), **tl.context(tensor))
    return gradient / size
