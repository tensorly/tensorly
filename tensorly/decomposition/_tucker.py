import numpy as np
from ..base import unfold
from ..tenalg import multi_mode_dot, mode_dot, norm
from ..tenalg import partial_svd
from ..tucker import tucker_to_tensor
from ..utils import check_random_state

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


def tucker(tensor, ranks=None, n_iter_max=100, init='svd', tol=10e-5,
           random_state=None, verbose=False):
    """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition:
        ``tensor = [| core; factors[0], ...factors[-1] |]``

    Parameters
    ----------
    tensor : ndarray
    ranks : None or int list
            size of the core tensor, ``(len(ranks) == tensor.ndim)``
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity

    Returns
    -------
    core : ndarray of size `ranks`
            core tensor of the Tucker decomposition
    factors : ndarray list
            list of factors of the Tucker decomposition.
            Its ``i``-th element is of shape ``(tensor.shape[i], ranks[i])``
    """
    if ranks is None:
        ranks = [s for s in tensor.shape]

    # SVD init
    if init == 'svd':
        factors = []
        for mode in range(tensor.ndim):
            eigenvecs, _, _ = partial_svd(unfold(tensor, mode), n_eigenvecs=ranks[mode])
            factors.append(eigenvecs)
    else:
        rng = check_random_state(random_state)
        core = rng.random_sample(ranks)
        factors = [rng.random_sample(s) for s in zip(tensor.shape, ranks)]

    rec_errors = []
    norm_tensor = norm(tensor, 2)

    for iteration in range(n_iter_max):
        for mode in range(tensor.ndim):
            core_approximation = tucker_to_tensor(tensor, factors, skip_factor=mode, transpose_factors=True)
            eigenvecs, _, _ = partial_svd(unfold(core_approximation, mode), n_eigenvecs=ranks[mode])
            factors[mode] = eigenvecs

        core = tucker_to_tensor(tensor, factors, transpose_factors=True)

        rec_error = np.sqrt(norm_tensor**2 - norm(core, 2)**2) / norm_tensor
        #rec_error = norm(tensor - tucker_to_tensor(core, factors), 2) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconsturction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break

    return core, factors


def non_negative_tucker(tensor, ranks, n_iter_max=10, init='svd', tol=10e-5,
                        random_state=None, verbose=False):
    """Non-negative Tucker decomposition

        Iterative multiplicative update, see [1]_, [2]_

    Parameters
    ----------
    tensor : ``ndarray``
    rank   : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}
    random_state : {None, int, np.random.RandomState}

    Returns
    -------
    core : ndarray
            positive core of the Tucker decomposition
            has shape `ranks`
    factors : ndarray list
            list of factors of the CP decomposition
            element `i` is of shape (tensor.shape[i], rank)

    References
    ----------
    .. [1] G. Zhou, A. Cichocki, Q. Zhao and S. Xie,
       "Nonnegative Matrix and Tensor Factorizations : An algorithmic perspective,"
       in IEEE Signal Processing Magazine, vol. 31, no. 3, pp. 54-65, May 2014.

    .. [2] DD Lee and HS Seung,
       "Algorithms for non-negative matrix factorization",
       in Advances in neural information processing systems (NIPS), 2001
    """
    epsilon = 10e-12

    # Initialisation
    if init == 'svd':
        core, factors = tucker(tensor, ranks)
        nn_factors = [np.abs(f) for f in factors]
        nn_core = np.abs(core)
    else:
        rng = check_random_state(random_state)
        core = rng.random_sample(ranks) + 0.01  # Check this
        factors = [rng.random_sample(s) for s in zip(tensor.shape, ranks)]
        nn_factors = [np.abs(f) for f in factors]
        nn_core = np.abs(core)

    n_factors = len(nn_factors)
    norm_tensor = norm(tensor, 2)
    rec_errors = []

    for iteration in range(n_iter_max):
        for mode in range(tensor.ndim):
            B = tucker_to_tensor(nn_core, nn_factors, skip_factor=mode)
            B = unfold(B, mode).T

            numerator = np.dot(unfold(tensor, mode), B)
            numerator = numerator.clip(min=epsilon)
            denominator = np.dot(nn_factors[mode], B.T.dot(B))
            denominator = denominator.clip(min=epsilon)
            nn_factors[mode] *= numerator / denominator

        numerator = tucker_to_tensor(tensor, nn_factors, transpose_factors=True)
        numerator = numerator.clip(min=epsilon)
        for i, f in enumerate(nn_factors):
            if i:
                denominator = mode_dot(denominator, f.T.dot(f), i)
            else:
                denominator = mode_dot(nn_core, f.T.dot(f), i)
        denominator = denominator.clip(min=epsilon)
        nn_core *= numerator / denominator

        rec_error = norm(tensor - tucker_to_tensor(nn_core, nn_factors), 2) / norm_tensor
        rec_errors.append(rec_error)
        if iteration > 1 and verbose:
            print('reconsturction error={}, variation={}.'.format(
                rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

        if iteration > 1 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print('converged in {} iterations.'.format(iteration))
            break

    return nn_core, nn_factors
