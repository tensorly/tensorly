""" Parallel Factor Analysis
"""
# Authors:  Jeremy E.Cohen
#           Jean Kossaifi

import warnings
import numpy as np

import tensorly as tl
from tensorly.random import check_random_state
from tensorly.base import unfold
from tensorly.kruskal_tensor import kruskal_to_tensor
from tensorly.tenalg import khatri_rao
from tensorly.contrib.optimization.optim_tensor_ls import least_squares_nway



def initialize_factors(tensor, rank, init='random', svd='numpy_svd', random_state=None, non_negative=False):
    """Initialize factors used in `parafac`.

    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.

    NOTES: NOT YET SET FOR CONSTRAINED OPTIMIZATION. THIS MEANS CONSTRAINED AND
    UNCONSTRAINED PARAFAC ARE HANDLED THE SAME WAY FOR INITIALIZATION.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD,
        acceptable values in tensorly.SVD_FUNS
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
            return [tl.abs(f) for f in factors]
        else:
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
            if non_negative:
                factors.append(tl.abs(U[:, :rank]))
            else:
                factors.append(U[:, :rank])
        return factors

    raise ValueError('Initialization method "{}" not recognized'.format(init))


def parafac_als(tensor, rank, init_factors, n_iter_max=100,
                tol=1e-8, verbose=False,
                return_errors=False,
                fixed_modes=[]):
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
    init_factors : list
        Table of initial factors. See initialize_factors().
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors


    Returns
    -------
    factors : ndarray list
        List of factors of the CP decomposition element `i` is of shape
        (tensor.shape[i], rank)
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """

    # initialisation
    factors = init_factors
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):

        # One pass of least squares on each updated mode
        factors, rec_error = least_squares_nway(tensor, factors,
                                                rank, norm_tensor,
                                                fixed_modes)

        if tol:
            rec_errors.append(rec_error)

            if iteration > 1:
                if verbose:
                    print('reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    if verbose:
                        print('converged in {} iterations.'.format(iteration))
                    break

    if return_errors:
        return factors, rec_errors
    else:
        return factors


class Parafac:
    """
    This is the doc
    """
    def __init__(self, rank=1, method='ALS', init='random', constraints=[],
                 weights=[], fixed_modes=[], n_iter_max=100, tol=1e-8,
                 verbose=False, svd='numpy_svd', random_state=None):
        """
        rank: Number of components in the model. Default is 1.

        init: if init is a string, then an initialisation method is used
        according to that string. If init is a set of factors, then those
        factors are used for initiatization. Empty init results in random
        initialization.

        Constraints: constraints are input as a table of strings, containing
        the type of constraint for each factor. For unconstrained
        decomposition, an empty constraints table can be used.

        weights: weighting of the entries of the input tensor. This is a tensor
            of the same size of the input data.

        fixed_modes: a vector containing the modes that are fixed by the user.
            These modes will not be updated when using Parafac.fit().
        """
        self.rank = rank
        self.method = method
        self.init = init
        self.constraints = constraints
        self.weights = weights
        self.fixed_modes = fixed_modes
        self.components = []
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.verbose = verbose
        self.svd = svd
        self.random_state = random_state

    def fit(self, data, init_factors=0):
        """
        intended behavior: call Parafac.fit(data) if init has been set for the
        Parafac instance,
        TODO:  call
        Parafac.fit(data,init) for testing various initializations.
        """
        # Call the initialisation method, depending on wether the user prodived
        # initial factors or initialization method.
        if init_factors == 0:
            if isinstance(self.init, str):
                # TODO cleanup nonnegative in init
                factors = initialize_factors(data, self.rank, self.init,
                                             svd=self.svd,
                                             non_negative=False,
                                             random_state=self.random_state)
            elif isinstance(self.init, list):
                factors = self.init
        else:
            factors = init_factors
        # Call the method
        if self.method == 'ALS':  # TODO change init_factors in parafac_als
            factors, errors = parafac_als(data, self.rank, factors,
                                          return_errors=True,
                                          n_iter_max=self.n_iter_max,
                                          tol=self.tol, verbose=self.verbose,
                                          fixed_modes=self.fixed_modes)
        elif self.method == 'CG':
            print('not implemented')
            # factors = parafac_cg()
        elif self.method == 'ADMM':
            print('not implemented')
            # factors = parafac_admm()
        elif self.method == 'HALS':
            print('not implemented')
            # factors = parafac_hals()

        # Filling in the components attribute and returning them
        self.components = factors
        return self.components, errors

    def reconstruct(self):
        """
        returns the resulting tensor of the fitted Parafac model
        """
        return kruskal_to_tensor(self.components)

        # Build the tensor using the Kruskal operator

    def transform(self, data, factors=0, rank=0):
        """
        transforms an input tensor by projecting it on the span of the
        estimated factors using the current Parafac model. This only makes
        sense if the rank is smaller than the dimensions. Other factors can be
        used if specified.
        """

        # For each mode where r<dim, project using the pseudo-inverse of each
        # factor
