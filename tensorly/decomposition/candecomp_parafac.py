import numpy as np
import warnings

import tensorly as tl
from ..random import check_random_state
from ..base import unfold
from ..kruskal_tensor import (kruskal_to_tensor, KruskalTensor,
                              unfolding_dot_khatri_rao, kruskal_norm)
from ..tenalg import khatri_rao

# Authors: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>
#          Chris Swierczewski <csw@amazon.com>
#          Sam Schneider <samjohnschneider@gmail.com>
#          Aaron Meurer <asmeurer@gmail.com>

# License: BSD 3 clause

def initialize_factors(tensor, rank, init='svd', svd='numpy_svd', random_state=None, 
                       non_negative=False, normalize_factors=False):
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
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned

    Returns
    -------
    factors : ndarray list
        List of initialized factors of the CP decomposition where element `i`
        is of shape (tensor.shape[i], rank)

    """
    rng = check_random_state(random_state)

    if init == 'random':
        factors = [tl.tensor(rng.random_sample((tensor.shape[i], rank)), **tl.context(tensor)) for i in range(tl.ndim(tensor))]
        if non_negative:
            factors = [tl.abs(f) for f in factors]
        if normalize_factors: 
            factors = [f/(tl.reshape(tl.norm(f, axis=0), (1, -1)) + 1e-12) for f in factors]
        return factors

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                    svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank)

            if tensor.shape[mode] < rank:
                # TODO: this is a hack but it seems to do the job for now
                # factor = tl.tensor(np.zeros((U.shape[0], rank)), **tl.context(tensor))
                # factor[:, tensor.shape[mode]:] = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                # factor[:, :tensor.shape[mode]] = U
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)
            
            factor = U[:, :rank]
            if non_negative:
                factor = tl.abs(factor)
            if normalize_factors:
                factor = factor / (tl.reshape(tl.norm(factor, axis=0), (1, -1)) + 1e-12)
            factors.append(factor)
        return factors

    raise ValueError('Initialization method "{}" not recognized'.format(init))


def parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd', normalize_factors=False,
            tol=1e-8, orthogonalise=False, random_state=None, verbose=0, return_errors=False,
            non_negative=False, mask=None):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [|weights; factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random'}, optional
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    non_negative : bool, optional
        Perform non_negative PARAFAC. See :func:`non_negative_parafac`.
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_


    Returns
    -------
    KruskalTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
            all ones if normalize_factors is False (default), 
            weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape
            (tensor.shape[i], rank)

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
       
    .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values." 
            Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.


    """
    epsilon = 10e-12

    if orthogonalise and not isinstance(orthogonalise, int):
        orthogonalise = n_iter_max

    factors = initialize_factors(tensor, rank, init=init, svd=svd,
                                 random_state=random_state,
                                 non_negative=non_negative,
                                 normalize_factors=normalize_factors)
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    weights = tl.ones(rank, **tl.context(tensor))

    for iteration in range(n_iter_max):
        if orthogonalise and iteration <= orthogonalise:
            factors = [tl.qr(f)[0] if min(tl.shape(f)) >= rank else f for i, f in enumerate(factors)]

        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in range(tl.ndim(tensor)):
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))
            if non_negative:
                accum = 1
                # khatri_rao(factors).tl.dot(khatri_rao(factors))
                # simplifies to multiplications
                sub_indices = [i for i in range(len(factors)) if i != mode]
                for i, e in enumerate(sub_indices):
                    if i:
                        accum *= tl.dot(tl.transpose(factors[e]), factors[e])
                    else:
                        accum = tl.dot(tl.transpose(factors[e]), factors[e])

            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse*tl.dot(tl.conj(tl.transpose(factor)), factor)

            if mask is not None:
                tensor = tensor*mask + tl.kruskal_to_tensor((None, factors), mask=1-mask)

            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

            if non_negative:
                numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
                denominator = tl.dot(factors[mode], accum)
                denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
                factor = factors[mode] * numerator / denominator
            else:
                factor = tl.transpose(tl.solve(tl.conj(tl.transpose(pseudo_inverse)),
                                      tl.transpose(mttkrp)))
            
            if normalize_factors:
                weights = tl.norm(factor, order=2, axis=0)
                weights = tl.where(tl.abs(weights) <= tl.eps(tensor.dtype), 
                                   tl.ones(tl.shape(weights), **tl.context(factors[0])),
                                   weights)
                factor = factor/(tl.reshape(weights, (1, -1)))

            factors[mode] = factor

        if tol:
            # ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>
            factors_norm = kruskal_norm((weights, factors))

            # mttkrp and factor for the last mode. This is equivalent to the
            # inner product <tensor, factorization>
            iprod = tl.sum(tl.sum(mttkrp*factor, axis=0)*weights)
            rec_error = tl.sqrt(tl.abs(norm_tensor**2 + factors_norm**2 - 2*iprod)) / norm_tensor
            rec_errors.append(rec_error)

            if iteration >= 1:
                if verbose:
                    print('reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    if verbose:
                        print('converged in {} iterations.'.format(iteration))
                    break       
            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    kruskal_tensor = KruskalTensor((weights, factors))

    if return_errors:
        return kruskal_tensor, rec_errors
    else:
        return kruskal_tensor


def non_negative_parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd',
                         tol=10e-7, random_state=None, verbose=0):
    """
    Non-negative CP decomposition

    Uses multiplicative updates, see [2]_

    This is the same as parafac(non_negative=True).

    Parameters
    ----------
    tensor : ndarray
    rank   : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
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
    return parafac(tensor, rank, n_iter_max=n_iter_max, init=init, svd=svd,
                   tol=tol, random_state=random_state, verbose=verbose, non_negative=True)


def sample_khatri_rao(matrices, n_samples, skip_matrix=None,
                      return_sampled_rows=False, random_state=None):
    """Random subsample of the Khatri-Rao product of the given list of matrices

        If one matrix only is given, that matrix is directly returned.

    Parameters
    ----------
    matrices : ndarray list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)

    n_samples : int
        number of samples to be taken from the Khatri-Rao product

    skip_matrix : None or int, optional, default is None
        if not None, index of a matrix to skip

    random_state : None, int or numpy.random.RandomState
        if int, used to set the seed of the random number generator
        if numpy.random.RandomState, used to generate random_samples

    returned_sampled_rows : bool, default is False
        if True, also returns a list of the rows sampled from the full
        khatri-rao product

    Returns
    -------
    sampled_Khatri_Rao : ndarray
        The sampled matricised tensor Khatri-Rao with `n_samples` rows

    indices : tuple list
        a list of indices sampled for each mode

    indices_kr : int list
        list of length `n_samples` containing the sampled row indices
    """
    if random_state is None or not isinstance(random_state, np.random.RandomState):
        rng = check_random_state(random_state)
        warnings.warn('You are creating a new random number generator at each call.\n'
                      'If you are calling sample_khatri_rao inside a loop this will be slow:'
                      ' best to create a rng outside and pass it as argument (random_state=rng).')
    else:
        rng = random_state

    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    rank = tl.shape(matrices[0])[1]
    sizes = [tl.shape(m)[0] for m in matrices]

    # For each matrix, randomly choose n_samples indices for which to compute the khatri-rao product
    indices_list = [rng.randint(0, tl.shape(m)[0], size=n_samples, dtype=int) for m in matrices]
    if return_sampled_rows:
        # Compute corresponding rows of the full khatri-rao product
        indices_kr = np.zeros((n_samples), dtype=int)
        for size, indices in zip(sizes, indices_list):
            indices_kr = indices_kr*size + indices

    # Compute the Khatri-Rao product for the chosen indices
    sampled_kr = tl.ones((n_samples, rank), **tl.context(matrices[0]))
    for indices, matrix in zip(indices_list, matrices):
        sampled_kr = sampled_kr*matrix[indices, :]

    if return_sampled_rows:
        return sampled_kr, indices_list, indices_kr
    else:
        return sampled_kr, indices_list


def randomised_parafac(tensor, rank, n_samples, n_iter_max=100, init='random', svd='numpy_svd',
                       tol=10e-9, max_stagnation=20, random_state=None, verbose=1):
    """Randomised CP decomposition via sampled ALS

    Parameters
    ----------
    tensor : ndarray
    rank   : int
            number of components
    n_samples : int
                number of samples per ALS step
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    max_stagnation: int, optional, default is 0
                    if not zero, the maximum allowed number
                    of iterations with no decrease in fit
    random_state : {None, int, np.random.RandomState}, default is None
    verbose : int, optional
        level of verbosity

    Returns
    -------
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``

    References
    ----------
    .. [3] Casey Battaglino, Grey Ballard and Tamara G. Kolda,
       "A Practical Randomized CP Tensor Decomposition",
    """
    rng = check_random_state(random_state)
    factors = initialize_factors(tensor, rank, init=init, svd=svd, random_state=random_state)
    rec_errors = []
    n_dims = tl.ndim(tensor)
    norm_tensor = tl.norm(tensor, 2)
    min_error = 0

    weights = tl.ones(rank, **tl.context(tensor))
    for iteration in range(n_iter_max):
        for mode in range(n_dims):
            kr_prod, indices_list = sample_khatri_rao(factors, n_samples, skip_matrix=mode, random_state=rng)
            indices_list = [i.tolist() for i in indices_list]
            # Keep all the elements of the currently considered mode
            indices_list.insert(mode, slice(None, None, None))
            # MXNet will not be happy if this is a list insteaf of a tuple
            indices_list = tuple(indices_list)
            if mode:
                sampled_unfolding = tensor[indices_list]
            else:
                sampled_unfolding = tl.transpose(tensor[indices_list])

            pseudo_inverse = tl.dot(tl.transpose(kr_prod), kr_prod)
            factor = tl.dot(tl.transpose(kr_prod), sampled_unfolding)
            factor = tl.transpose(tl.solve(pseudo_inverse, factor))
            factors[mode] = factor

        if max_stagnation or tol:
            rec_error = tl.norm(tensor - kruskal_to_tensor((weights, factors)), 2) / norm_tensor
            if not min_error or rec_error < min_error:
                min_error = rec_error
                stagnation = -1
            stagnation += 1

            rec_errors.append(rec_error)

            if iteration > 1:
                if verbose:
                    print('reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                if (tol and abs(rec_errors[-2] - rec_errors[-1]) < tol) or \
                   (stagnation and (stagnation > max_stagnation)):
                    if verbose:
                        print('converged in {} iterations.'.format(iteration))
                    break

    return KruskalTensor((weights, factors))
