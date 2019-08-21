import tensorly as tl
from ..base import unfold
from ..tenalg import multi_mode_dot, mode_dot
from ..tucker_tensor import tucker_to_tensor
from ..random import check_random_state
from math import sqrt

import warnings

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>

# License: BSD 3 clause

def partial_tucker(tensor, modes, rank=None, n_iter_max=100, init='svd', tol=10e-5,
                   svd='numpy_svd', random_state=None, verbose=False, ranks=None):
    """Partial tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition exclusively along the provided modes.

    Parameters
    ----------
    tensor : ndarray
    modes : int list
            list of the modes on which to perform the decomposition
    ranks : None or int list
            size of the core tensor, ``(len(ranks) == len(modes))``
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD,
        acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity

    Returns
    -------
    core : ndarray 
            core tensor of the Tucker decomposition
    factors : ndarray list
            list of factors of the Tucker decomposition.
            with ``core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes``

    """
    if ranks is not None:
        message = "'ranks' is depreciated, please use 'rank' instead"
        warnings.warn(message, DeprecationWarning)
        rank = ranks

    if rank is None:
        message = "No value given for 'rank'. The decomposition will preserve the original size."
        warnings.warn(message, Warning)
        rank = [tl.shape(tensor)[mode] for mode in modes]
    elif isinstance(rank, int):
        message = "Given only one int for 'rank' intead of a list of {} modes. Using this rank for all modes.".format(len(modes))
        warnings.warn(message, Warning)
        rank = [rank for _ in modes]

    try:
        svd_fun = tl.SVD_FUNS[svd]
    except KeyError:
        message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                svd, tl.get_backend(), tl.SVD_FUNS)
        raise ValueError(message)

    # SVD init
    if init == 'svd':
        factors = []
        for index, mode in enumerate(modes):
            eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank[index])
            factors.append(eigenvecs)
    else:
        rng = check_random_state(random_state)
        core = tl.tensor(rng.random_sample(rank), **tl.context(tensor))
        factors = [tl.tensor(rng.random_sample((tl.shape(tensor)[mode], rank[index])), **tl.context(tensor)) for (index, mode) in enumerate(modes)]

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):
        for index, mode in enumerate(modes):
            core_approximation = multi_mode_dot(tensor, factors, modes=modes, skip=index, transpose=True)
            eigenvecs, _, _ = svd_fun(unfold(core_approximation, mode), n_eigenvecs=rank[index])
            factors[index] = eigenvecs

        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)

        # The factors are orthonormal and therefore do not affect the reconstructed tensor's norm
        rec_error = sqrt(abs(norm_tensor**2 - tl.norm(core, 2)**2)) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconstruction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break

    return core, factors


def tucker(tensor, rank=None, ranks=None, n_iter_max=100, init='svd',
           svd='numpy_svd', tol=10e-5, random_state=None, verbose=False):
    """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition:
        ``tensor = [| core; factors[0], ...factors[-1] |]`` [1]_

    Parameters
    ----------
    tensor : ndarray
    ranks : None or int list
            size of the core tensor, ``(len(ranks) == tensor.ndim)``
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD,
        acceptable values in tensorly.SVD_FUNS
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

    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    modes = list(range(tl.ndim(tensor)))
    return partial_tucker(tensor, modes, rank=rank, ranks=ranks, n_iter_max=n_iter_max, init=init,
                          svd=svd, tol=tol, random_state=random_state, verbose=verbose)


def non_negative_tucker(tensor, rank, n_iter_max=10, init='svd', tol=10e-5,
                        random_state=None, verbose=False, ranks=None):
    """Non-negative Tucker decomposition

        Iterative multiplicative update, see [2]_

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
            element `i` is of shape ``(tensor.shape[i], rank)``

    References
    ----------
    .. [2] Yong-Deok Kim and Seungjin Choi,
       "Nonnegative tucker decomposition",
       IEEE Conference on Computer Vision and Pattern Recognition s(CVPR),
       pp 1-8, 2007
    """
    if ranks is not None:
        message = "'ranks' is depreciated, please use 'rank' instead"
        warnings.warn(message, DeprecationWarning)
        rank = ranks

    if rank is None:
        rank = [tl.shape(tensor)[mode] for mode in range(tl.ndim(tensor))]

    elif isinstance(rank, int):
        n_mode = tl.ndim(tensor)
        message = "Given only one int for 'rank' for decomposition a tensor of order {}. Using this rank for all modes.".format(n_mode)
        warnings.warn(message, RuntimeWarning)
        rank = [rank]*n_mode


    epsilon = 10e-12

    # Initialisation
    if init == 'svd':
        core, factors = tucker(tensor, rank)
        nn_factors = [tl.abs(f) for f in factors]
        nn_core = tl.abs(core)
    else:
        rng = check_random_state(random_state)
        core = tl.tensor(rng.random_sample(rank) + 0.01, **tl.context(tensor))  # Check this
        factors = [tl.tensor(rng.random_sample(s), **tl.context(tensor)) for s in zip(tl.shape(tensor), rank)]
        nn_factors = [tl.abs(f) for f in factors]
        nn_core = tl.abs(core)

    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []

    for iteration in range(n_iter_max):
        for mode in range(tl.ndim(tensor)):
            B = tucker_to_tensor((nn_core, nn_factors), skip_factor=mode)
            B = tl.transpose(unfold(B, mode))

            numerator = tl.dot(unfold(tensor, mode), B)
            numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
            denominator = tl.dot(nn_factors[mode], tl.dot(tl.transpose(B), B))
            denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
            nn_factors[mode] *= numerator / denominator

        numerator = tucker_to_tensor((tensor, nn_factors), transpose_factors=True)
        numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
        for i, f in enumerate(nn_factors):
            if i:
                denominator = mode_dot(denominator, tl.dot(tl.transpose(f), f), i)
            else:
                denominator = mode_dot(nn_core, tl.dot(tl.transpose(f), f), i)
        denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
        nn_core *= numerator / denominator

        rec_error = tl.norm(tensor - tucker_to_tensor((nn_core, nn_factors)), 2) / norm_tensor
        rec_errors.append(rec_error)
        if iteration > 1 and verbose:
            print('reconstruction error={}, variation={}.'.format(
                rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

        if iteration > 1 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print('converged in {} iterations.'.format(iteration))
            break

    return nn_core, nn_factors
