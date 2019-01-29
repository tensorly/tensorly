""" Parallel Factor Analysis
TODO : 
- nnls for nonnegative factors
- precompute unfoldings
- admm for (sparse, nonnegative/box constraints... (search)). Allow
  personnalization of constraints?
- line search for ALS
"""
# Authors:  Jeremy E.Cohen
#           Jean Kossaifi

import numpy as np
import tensorly as tl
from tensorly.random import check_random_state
from tensorly.base import unfold
from tensorly.kruskal_tensor import kruskal_to_tensor
from tensorly.contrib.optimization.optim_routines import least_squares_nway
from tensorly.contrib.optimization.optim_routines import fast_gradient_step
from tensorly.contrib.optimization.optim_routines import multiplicative_update_step


def initialize_factors(tensor, rank, init='random',
                       svd='numpy_svd', random_state=None):
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

    Returns
    -------
    factors : ndarray list
        List of initialized factors of the CP decomposition where element `i`
        is of shape (tensor.shape[i], rank)

    """
    rng = check_random_state(random_state)

    if init == 'random':
        factors = [tl.tensor(rng.random_sample((tensor.shape[i], rank)), **tl.context(tensor)) for i in range(tl.ndim(tensor))]
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
            factors.append(U[:, :rank])
        return factors

    raise ValueError('Initialization method "{}" not recognized'.format(init))


def parafac_als(tensor, rank, factors, n_iter_max=100,
                tol=1e-8, verbose=False,
                return_errors=False,
                fixed_modes=[]):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
        Input data tensor
    rank  : int
        Number of components.
    factors : list
        Table of initial factors. See initialize_factors(), or the
        initialize_parafac() method of class parafac.
    n_iter_max : int
        Maximum number of iteration
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        (Default: False) Level of verbosity
    return_errors : bool, optional
        (Default: False) Activate return of iteration errors
    fixed_modes : list, optional
        (Default: []) List of components indexes that are not updates. Returned
        values for these indexes are therefore the initialization values.


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


def parafac_mu(tensor, rank, factors, n_iter_max=100,
               tol=1e-8, verbose=False,
               return_errors=False,
               fixed_modes=[], epsilon=1e-12):
    """Nonnegative CANDECOMP/PARAFAC decomposition via
    multiplicative updates [2].

    Computes a rank-`rank` decomposition of `tensor` such that,

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
        Input data tensor
    rank  : int
        Number of components.
    factors : list
        Table of initial factors. See initialize_factors(), or the
        initialize_parafac() method of class parafac.
    n_iter_max : int
        Maximum number of iteration
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        (Default: False) Level of verbosity
    return_errors : bool, optional
        (Default: False) Activate return of iteration errors
    fixed_modes : list, optional
        (Default: []) List of components indexes that are not updates. Returned
        values for these indexes are therefore the initialization values.
    epsilon : float, optional
        (Default: 1e-12) Regularization term to avoid division by zero in
        multiplicative update.

    Returns
    -------
    factors : ndarray list
        List of factors of the CP decomposition element `i` is of shape
        (tensor.shape[i], rank). Factors are nonnegative.
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
   .. [2] Amnon Shashua and Tamir Hazan,
       "Non-negative tensor factorization with applications to statistics and computer vision",
       In Proceedings of the International Conference on Machine Learning (ICML),
       pp 792-799, ICML, 2005
    """

    # initialisation: project variables if negative
    for mode in range(tl.ndim(tensor)):
        factors[mode][factors[mode] < 0] = 0
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):

        # One pass of least squares on each updated mode
        factors, rec_error = multiplicative_update_step(tensor, factors,
                                                        rank, norm_tensor,
                                                        fixed_modes,
                                                        epsilon)

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


def parafac_fg(tensor, rank, factors, n_iter_max=100,
               tol=1e-8, verbose=False,
               return_errors=False, step = 1e-5,
               fixed_modes=[], alpha = 0.2,
               qstep = 0, constraints = None, weights = None):
    """CANDECOMP/PARAFAC decomposition via proximal fast gradient (FG)

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
        Input data tensor
    rank  : int
        Number of components.
    factors : list
        Table of initial factors. See initialize_factors(), or the
        initialize_parafac() method of class parafac.
    n_iter_max : int
        Maximum number of iteration
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        (Default: False) Level of verbosity
    return_errors : bool, optional
        (Default: False) Activate return of iteration errors
    step : float, stepsize, optional (tuning is recommended)
        (Default: 1e-5) How much the gradient is descended in the FG method.
        Choose 1/L where L is the Lipschitz constant if possible.
    fixed_modes : list, optional
        (Default: []) List of components indexes that are not updates. Returned
        values for these indexes are therefore the initialization values.
    alpha : float in ]0,1[, optional
        (Default: 0.2) Initial value of the alpha parameter in Nesterov
        acceleration.
    qstep : float in ]0,step], optional
        (Default: 0) A constant in the Nesterov acceleration scheme. If solving
        a least squares problem, use the smallest eigenvalue of the mixing
        matrix.
    constraints : list of strings, optional
        (Default: []) Contains keywords for imposing constraints on each
        factor. For a full list of available constraints, check *** TODO.
        Ex: for nonnegative factorization with a third order tensor, use
        constraints=['NN','NN','NN'].

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
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    aux_factors = np.copy(factors)

    for iteration in range(n_iter_max):

        # One pass of fast gradient
        factors, rec_error = fast_gradient_step(tensor, factors, rank, norm_tensor,
                                                aux_factors, step, alpha, qstep,
                                                fixed_modes, weights, constraints)

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
    A class for defining a Parafac model and an optimization technique.
    Computing PARAFAC of a tensor means finding factor matrices so that

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    This class allows to choose the optimization method, the initialization
    method, constraints on each mode, fixed modes, and various optimization
    parameters such as the number of iterations, gradient steps and so on.


    Atributes
    ---------

    Methods
    -------

    """

    def __init__(self, rank=1, method='ALS', init='random', constraints=[],
                 weights=[], fixed_modes=[], n_iter_max=100, tol=1e-8,
                 verbose=False, svd='numpy_svd', random_state=None,
                 step=1e-5, alpha=0.2, epsilon=1e-12):
        """
        TODO : docstring

        rank: Number of components in the model. Default is 1.

        init: string, refers to initialization technique.  To initialize with a
        set of factors, use init_factors(data, factors).

        constraints: constraints are input as a table of strings, containing
        the type of constraint for each factor. For unconstrained
        decomposition, an empty constraints table can be used.
            ['NN', x, x] : first factor is nonnegative

        method: the optimization routine used to compute the PARAFAC model.
        Defaults are 'ALS' for unconstrained optimization, and 'Fast Gradient'
        for constrained optimization or in the presence of missing data.

        weights: weighting of the entries of the input tensor. This is a tensor
            of the same size of the input data.

        fixed_modes: a vector containing the modes that are fixed by the user.
            These modes will not be updated when using Parafac.fit().
        """
        self.rank = rank
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
        self.step = step
        self.alpha = alpha
        self.method = method
        self.init_factors = 0
        self.epsilon = epsilon

        # Choosing the default method based on the constraints
        if not constraints:
            self.method = method
        elif constraints and method == 'ALS':
            print('Warning: ALS does not apply to constrained Parafac,')
            print(' using default projected fast gradient instead.')
            self.method = 'FG'

    def initialize_parafac(self, data=0, init_factors=0):
        """
        Compute initial factors for the Parafac model, stored in attribute
        self.init_factors
        
        Parameters
        ----------

        data : ndarray, optional
            Input tensor
        init_factors : list of arrays, optional
            Initial factors. This initialises the model instance using a priori
            known factors, obtained by the user using a method of his own.
        """
        # If only the data is given
        if init_factors == 0:
            self.init_factors = initialize_factors(data, self.rank, self.init,
                                                   svd=self.svd,
                                                   random_state=self.random_state)
        else:  # if factors are provided
            self.init_factors = init_factors

    def fit(self, data):
        """
        Parafac.fit(data) for testing various initializations.
        """
        # Initialize factors
        self.initialize_parafac(data, self.init_factors)
        factors = np.copy(self.init_factors)

        # Call the method
        if self.method == 'ALS':
            factors, errors = parafac_als(data, self.rank, factors,
                                          return_errors=True,
                                          n_iter_max=self.n_iter_max,
                                          tol=self.tol, verbose=self.verbose,
                                          fixed_modes=self.fixed_modes)
        elif self.method == 'FG':
            factors, errors = parafac_fg(data, self.rank, factors,
                                         return_errors=True,
                                         n_iter_max=self.n_iter_max,
                                         tol=self.tol, verbose=self.verbose,
                                         fixed_modes=self.fixed_modes,
                                         step = self.step, alpha = self.alpha,
                                         constraints=self.constraints)
        elif self.method == 'MU':
            # Checking that all constraints are set to 'NN'
            for i in range(tl.ndim(data)):
                if self.constraints[i] != 'NN':
                    print('Warning: MU will only perform nonnegative PARAFAC.')
                    print('For unconstrained PARAFAC, use ALS instead.')
            factors, errors = parafac_mu(data, self.rank, factors,
                                         return_errors=True,
                                         n_iter_max=self.n_iter_max,
                                         tol=self.tol, verbose=self.verbose,
                                         fixed_modes=self.fixed_modes,
                                         epsilon=self.epsilon)

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
