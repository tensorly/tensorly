import numpy as np

from .. import backend as T
from ..random import check_random_state
from ..base import unfold
from ..kruskal_tensor import kruskal_to_tensor
from ..tenalg import khatri_rao

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>
# Author: Chris Swierczewski <csw@amazon.com>

# License: BSD 3 clause


def normalize_factors(factors):
    """Normalizes factors to unit length and returns factor magnitudes

    Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,
    where the columns of each `V_k` are normalized to unit Euclidean length
    from the columns of `U_k` with the normalizing constants absorbed into
    `weights`. In the special case of a symmetric tensor, `weights` holds the
    eigenvalues of the tensor.

    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        i.e.::
            for u in U:
                u[i].shape == (s_i, R)

        where `R` is fixed while `s_i` can vary with `i`

    Returns
    -------
    normalized_factors : list of ndarrays
        list of matrices with the same shape as `factors`
    weights : ndarray
        vector of length `R` holding normalizing constants

    """
    # allocate variables for weights, and normalized factors
    rank = factors[0].shape[1]
    weights = T.ones(rank)
    normalized_factors = []

    # normalize columns of factor matrices
    for factor in factors:
        scales = T.norm(factor, axis=0)
        weights *= scales
        scales_non_zero = T.where(scales==0, T.ones(T.shape(scales)), scales)
        normalized_factors.append(factor/scales_non_zero)
    return normalized_factors, weights


def initialize_factors(tensor, rank, init='svd', random_state=None):
    r"""Initialize factors used in `parafac`.

    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random'}, optional

    Returns
    -------
    factors : ndarray list
        List of initialized factors of the CP decomposition where element `i`
        is of shape (tensor.shape[i], rank)

    """
    rng = check_random_state(random_state)

    if init is 'random':
        factors = [T.tensor(rng.random_sample((tensor.shape[i], rank)), **T.context(tensor)) for i in range(T.ndim(tensor))]
        return factors
    elif init is 'svd':
        factors = []
        for mode in range(T.ndim(tensor)):
            U, _, _ = T.partial_svd(unfold(tensor, mode), n_eigenvecs=rank)

            if tensor.shape[mode] < rank:
                # TODO: this is a hack but it seems to do the job for now
                # factor = T.tensor(np.zeros((U.shape[0], rank)), **T.context(tensor))
                # factor[:, tensor.shape[mode]:] = T.tensor(rng.random_sample((U.shape[0], rank - T.shape(tensor)[mode])), **T.context(tensor))
                # factor[:, :tensor.shape[mode]] = U
                random_part = T.tensor(rng.random_sample((U.shape[0], rank - T.shape(tensor)[mode])), **T.context(tensor))
                U = T.concatenate([U, random_part], axis=1)
            factors.append(U[:, :rank])
        return factors

    raise ValueError('Initialization method "{}" not recognized'.format(init))


def parafac(tensor, rank, n_iter_max=100, init='svd', tol=1e-7,
            random_state=None, verbose=False):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random'}, optional
        Type of factor matrix initialization. See `initialize_factors`.
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity


    Returns
    -------
    factors : ndarray list
        List of factors of the CP decomposition element `i` is of shape
        (tensor.shape[i], rank)
    weights : ndarray, optional
        Array of length `rank` of weights for each factor matrix. See the
        `with_weights` keyword attribute.
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    factors = initialize_factors(tensor, rank, init=init, random_state=random_state)
    rec_errors = []
    norm_tensor = T.norm(tensor, 2)

    for iteration in range(n_iter_max):
        for mode in range(T.ndim(tensor)):
            pseudo_inverse = T.tensor(np.ones((rank, rank)), **T.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse*T.dot(T.transpose(factor), factor)
            factor = T.dot(unfold(tensor, mode), khatri_rao(factors, skip_matrix=mode))
            factor = T.transpose(T.solve(T.transpose(pseudo_inverse), T.transpose(factor)))
            factors[mode] = factor

        #if verbose or tol:
        rec_error = T.norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor
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

        Uses multiplicative updates, see [2]_

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
            element `i` is of shape ``(tensor.shape[i], rank)``

    References
    ----------
    .. [2] Amnon Shashua and Tamir Hazan,
       "Non-negative tensor factorization with applications to statistics and computer vision",
       In Proceedings of the International Conference on Machine Learning (ICML),
       pp 792-799, ICML, 2005
    """
    epsilon = 10e-12

    # Initialisation
    if init == 'svd':
        factors = parafac(tensor, rank)
        nn_factors = [T.abs(f) for f in factors]
    else:
        rng = check_random_state(random_state)
        nn_factors = [T.tensor(np.abs(rng.random_sample((s, rank))), **T.context(tensor)) for s in tensor.shape]

    n_factors = len(nn_factors)
    norm_tensor = T.norm(tensor, 2)
    rec_errors = []

    for iteration in range(n_iter_max):
        for mode in range(T.ndim(tensor)):
            # khatri_rao(factors).T.dot(khatri_rao(factors))
            # simplifies to multiplications
            sub_indices = [i for i in range(n_factors) if i != mode]
            for i, e in enumerate(sub_indices):
                if i:
                    accum = accum*T.dot(T.transpose(nn_factors[e]), nn_factors[e])
                else:
                    accum = T.dot(T.transpose(nn_factors[e]), nn_factors[e])

            numerator = T.dot(unfold(tensor, mode), khatri_rao(nn_factors, skip_matrix=mode))
            numerator = T.clip(numerator, a_min=epsilon, a_max=None)
            denominator = T.dot(nn_factors[mode], accum)
            denominator = T.clip(denominator, a_min=epsilon, a_max=None)
            nn_factors[mode] = nn_factors[mode]* numerator / denominator

        rec_error = T.norm(tensor - kruskal_to_tensor(nn_factors), 2) / norm_tensor
        rec_errors.append(rec_error)
        if iteration > 1 and verbose:
            print('reconstruction error={}, variation={}.'.format(
                rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

        if iteration > 1 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print('converged in {} iterations.'.format(iteration))
            break

    return nn_factors
