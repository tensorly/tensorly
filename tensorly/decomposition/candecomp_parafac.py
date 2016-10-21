import numpy as np
from numpy.linalg import solve
from ..utils import check_random_state
from ..base import unfold
from ..kruskal import kruskal_to_tensor
from ..tenalg import khatri_rao
from ..tenalg._partial_svd import partial_svd
from ..tenalg import norm

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


def parafac(tensor, rank, n_iter_max=100, init='svd', tol=10e-7,
            random_state=None, verbose=False):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)

        Computes a rank-`rank` decomposition of `tensor` [1]_ such that:
        ``tensor = [| factors[0], ..., factors[-1] |]``

    Parameters
    ----------
    tensor : ndarray
    rank  : int
            number of components
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
    factors : ndarray list
            list of factors of the CP decomposition
            element `i` is of shape (tensor.shape[i], rank)

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    tensor = tensor.astype(np.float)
    rng = check_random_state(random_state)
    if init is 'random':
        factors = [rng.random_sample((tensor.shape[i], rank)) for i in range(tensor.ndim)]

    elif init is 'svd':
        factors = []
        for mode in range(tensor.ndim):
            U, _, _ = partial_svd(unfold(tensor, mode), n_eigenvecs=rank)

            if tensor.shape[mode] < rank:
                # TODO: this is a hack but it seems to do the job for now
                new_columns = rng.random_sample((U.shape[0], rank - tensor.shape[mode]))
                U = np.hstack((U, new_columns))
            factors.append(U[:, :rank])

    rec_errors = []
    norm_tensor = norm(tensor, 2)

    for iteration in range(n_iter_max):
        for mode in range(tensor.ndim):
            pseudo_inverse = np.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse *= np.dot(factor.T, factor)
            factor = np.dot(unfold(tensor, mode), khatri_rao(factors, skip_matrix=mode))
            factor = solve(pseudo_inverse.T, factor.T).T
            factors[mode] = factor

        #if verbose or tol:
        rec_error = norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconsturction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break

    return factors


def non_negative_parafac(tensor, rank, n_iter_max=100, init='svd', tol=10e-7,
                         random_state=None, verbose=0):
    """Non-negative CP decomposition

        Uses multiplicative updates, see [1]_, [2]_

    Parameters
    ----------
    tensor : ndarray
    rank   : int
            number of components
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
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape `(tensor.shape[i], rank)`

    References
    ----------
    .. [1] DD Lee and HS Seung,
       "Algorithms for non-negative matrix factorization",
       in Advances in neural information processing systems (NIPS), 2001

    .. [2] G. Zhou, A. Cichocki, Q. Zhao and S. Xie,
       "Nonnegative Matrix and Tensor Factorizations : An algorithmic perspective",
       in IEEE Signal Processing Magazine, vol. 31, no. 3, pp. 54-65, May 2014.
    """
    epsilon = 10e-12

    # Initialisation
    if init == 'svd':
        factors = parafac(tensor, rank)
        nn_factors = [np.abs(f) for f in factors]
    else:
        rng = check_random_state(random_state)
        nn_factors = [np.abs(rng.random_sample((s, rank))) for s in tensor.shape]

    n_factors = len(nn_factors)
    norm_tensor = norm(tensor, 2)
    rec_errors = []

    for iteration in range(n_iter_max):
        for mode in range(tensor.ndim):
            # khatri_rao(factors).T.dot(khatri_rao(factors))
            # simplifies to multiplications
            sub_indices = [i for i in range(n_factors) if i != mode]
            for i, e in enumerate(sub_indices):
                if i:
                    accum *= nn_factors[e].T.dot(nn_factors[e])
                else:
                    accum = nn_factors[e].T.dot(nn_factors[e])

            numerator = np.dot(unfold(tensor, mode), khatri_rao(nn_factors, skip_matrix=mode))
            numerator = numerator.clip(min=epsilon)
            denominator = np.dot(nn_factors[mode], accum)
            denominator = denominator.clip(min=epsilon)
            nn_factors[mode] *= numerator / denominator

        rec_error = norm(tensor - kruskal_to_tensor(nn_factors), 2) / norm_tensor
        rec_errors.append(rec_error)
        if iteration > 1 and verbose:
            print('reconstruction error={}, variation={}.'.format(
                rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

        if iteration > 1 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print('converged in {} iterations.'.format(iteration))
            break

    return nn_factors
