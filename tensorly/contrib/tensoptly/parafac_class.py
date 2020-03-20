""" Parallel Factor Analysis
TODO :
- better init
- admm for (sparse, nonnegative/box constraints... (search)). Allow
  personnalization of constraints?
- line search for ALS --> Extrapolation
"""
# Authors:  Jeremy E.Cohen
#           Axel Marmoret
#           Jean Kossaifi

import tensorly as tl
from ...random import check_random_state
from ...base import unfold
from ...kruskal_tensor import kruskal_to_tensor
from .parafac_routines import one_step_hals
from .parafac_routines import one_step_als
from .parafac_routines import fast_gradient_step
from .parafac_routines import multiplicative_update_step


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
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)
            factors.append(U[:, :rank])
        return factors

    raise ValueError('Initialization method "{}" not recognized'.format(init))


def bcd(tensor, rank, in_factors, method,
                n_iter_max=100,
                tol=1e-8, verbose=False,
                return_errors=True,
                constraints=None, fixed_modes=[],
                alpha=1, delta=0,
                epsilon=1e-12):
    """CANDECOMP/PARAFAC decomposition via block-coordinate descent (BCD)

    Computes a (constrained) rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
        Input data tensor
    rank  : int
        Number of components.
    in_factors : list
        Table of initial factors. See initialize_factors(), or the
        initialize_parafac() method of class parafac.
    method : string, default is 'ALS'
        Defines the otimization algorithm within the class of BCD methods to compute the decomposition.
        Some methods are not adapted for specific constraints, a warning message will be printed.
    n_iter_max : int
        Maximum number of iteration
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        (Default: False) Level of verbosity
    return_errors : bool, optional
        (Default: False) Activate return of iteration errors.
    constraints : list of strings, optional
        (Default: None) A list of predefined constraints on each mode.
    fixed_modes : list, optional
        (Default: []) List of components indexes that are not updates. Returned
        values for these indexes are therefore the initialization values.
    alpha, delta : float
        (Default: 1, 0) Parameters for accHALS.
    epsilon : float, optional
        (Default: 1e-12) Regularization term to avoid division by zero in
        multiplicative update.


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
    .. [2] Amnon Shashua and Tamir Hazan,
       "Non-negative tensor factorization with applications to statistics and computer vision",
       In Proceedings of the International Conference on Machine Learning (ICML),
       pp 792-799, ICML, 2005
    .. [3]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
        Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
        Neural Computation 24 (4): 1085-1105, 2012.
    """

    # initialisation
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    factors = in_factors.copy()

    for iteration in range(n_iter_max):

        # Generating the mode update sequence
        gen = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]
        compute_error = False  # bugged?

        for mode in gen:
            # Computing the rec error only on last mode
            if mode == gen[-1]:
                compute_error = True
            else:
                compute_error = False

            if method == 'ALS':
                # Changing factors inplace
                rec_error = one_step_als(tensor, factors,
                                         mode, rank, norm_tensor,
                                         compute_error)

            if method == 'HALS':
                rec_error = one_step_hals(tensor, factors, mode, rank,
                                          norm_tensor, compute_error,
                                          constraints, alpha=1, delta=0)

            if method == 'MU':
                rec_error = multiplicative_update_step(tensor, factors,
                               mode, rank, norm_tensor, fixed_modes,
                               epsilon, compute_error)

        if tol:
            # Error storred only for the last updated mode.
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


def aao(tensor, rank, in_factors, method,
        n_iter_max=100, tol=1e-8, verbose=False,
        return_errors=False, step=1e-5,
        fixed_modes=[], alpha=0.2,
        qstep=0, constraints=None, weights=None):
    """ All-at-once optimization for computing PARAFAC.

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
        Input data tensor
    rank  : int
        Number of components.
    in_factors : list
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
    factors = tl.copy(in_factors)
    aux_factors = tl.copy(factors)

    for iteration in range(n_iter_max):

        if method == 'FG':
            # One pass of fast gradient
            rec_error = fast_gradient_step(tensor, factors, rank, norm_tensor,
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
                 weights=None, fixed_modes=[], n_iter_max=100, tol=1e-8,
                 verbose=False, svd='numpy_svd', random_state=None,
                 step=1e-5, alpha=0.2, epsilon=1e-12,
                 init_factors=0, alpha_hals=1, delta_hals=0):
        """
        TODO : docstring

        rank: Number of components in the model. Default is 1.

        init: string, refers to initialization technique.  To initialize with a
        set of factors, use init_factors(data, factors).

        constraints: constraints are input as a table of strings, containing
        the type of constraint for each factor. For unconstrained
        decomposition, an empty constraints table can be used.
            ['NN', x, x]: first factor is nonnegative
            ** dev **
            ['DIC', x, x]: first factor is constrained to be in the columns of
            a dictionary, provided with input dico=D

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
        self.init_factors = init_factors
        self.epsilon = epsilon
        self.alpha_hals = alpha_hals
        self.delta_hals = delta_hals

        # Choosing the default method based on the constraints
        if not constraints:
            self.method = method
        elif constraints and method == 'ALS':
            print('Warning: ALS does not apply to constrained Parafac,')
            print(' using default projected fast gradient instead.')
            self.method = 'FG'

    def initialize_parafac(self, data=0):
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
        if self.init_factors == 0:
            self.init_factors = initialize_factors(data, self.rank, self.init,
                                                   svd=self.svd,
                                                   random_state=self.random_state)

    def fit(self, data):
        """
        Parafac.fit(data) for testing various initializations.
        """
        # Initialize factors
        self.initialize_parafac(data=data)

        # Call the method
        if self.method == 'ALS':
            print('Using ALS to compute Parafac')
            factors, errors = bcd(data, self.rank, self.init_factors, 'ALS', return_errors=True, n_iter_max=self.n_iter_max, tol=self.tol, verbose=self.verbose, fixed_modes=self.fixed_modes)

        elif self.method == 'HALS':
            for i in range(tl.ndim(data)):
                if self.constraints[i] != 'NN':
                    print('Warning: HALS only performs nonnegative PARAFAC.')
                    print('For unconstrained modes, ALS used instead.')
            factors, errors = bcd(data, self.rank, self.init_factors,
                                  'HALS', return_errors=True,
                                  n_iter_max=self.n_iter_max,
                                  tol=self.tol, verbose=self.verbose,
                                  fixed_modes=self.fixed_modes,
                                  constraints=self.constraints,
                                  alpha=self.alpha_hals,
                                  delta=self.delta_hals)

        elif self.method == 'MU':
            # Checking that all constraints are set to 'NN'
            for i in range(tl.ndim(data)):
                 if self.constraints[i] != 'NN':
                     print('Warning: MU will only perform nonnegative PARAFAC.')
                     print('For unconstrained PARAFAC, use ALS instead.')
            factors, errors = bcd(data, self.rank, self.init_factors, 'MU',
                                         return_errors=True,
                                         n_iter_max=self.n_iter_max,
                                         tol=self.tol, verbose=self.verbose,
                                         fixed_modes=self.fixed_modes,
                                         epsilon=self.epsilon)

        elif self.method == 'FG':
            factors, errors = aao(data, self.rank, self.init_factors, 'FG',
                                  return_errors=True,
                                  n_iter_max=self.n_iter_max,
                                  tol=self.tol, verbose=self.verbose,
                                  fixed_modes=self.fixed_modes,
                                  step=self.step, alpha=self.alpha,
                                  constraints=self.constraints)

        elif self.method == 'ADMM':
            print('not implemented')
            # factors = parafac_admm()

        # Filling in the components attribute and returning them
        # TODO: output in kruskal_tensor class?
        # For now, components and weights are attributes
        self.components = factors
        return self.components, errors

    def normalize(self):
        """
        Normalizes the components column-wise and returns the weights in self.weights.
        """
        weights = tl.ones(self.rank)  # TODO **tl.context(self.tensor))
        for (i, factor) in enumerate(self.components):
            norms = tl.norm(factor, order=2, axis=0)
            weights = weights*norms
            self.components[i] = factor/norms
        self.weights = weights

    def reconstruct(self):
        """
        returns the resulting tensor of the fitted Parafac model
        """
        return kruskal_to_tensor((None, self.components))

    def transform(self, data, factors=0, rank=0):
        """
        transforms an input tensor by projecting it on the span of the
        estimated factors using the current Parafac model. This only makes
        sense if the rank is smaller than the dimensions. Other factors can be
        used if specified.
        """
        # For each mode where r<dim, project using the pseudo-inverse of each
        # factor
        # ??
