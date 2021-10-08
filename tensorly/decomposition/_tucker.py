import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..base import unfold
from ..tenalg import multi_mode_dot, mode_dot
from ..tucker_tensor import tucker_to_tensor, TuckerTensor, validate_tucker_rank, tucker_normalize
import tensorly.tenalg as tlg
from ..tenalg.proximal import hals_nnls, active_set_nnls, fista
from math import sqrt
import warnings
from collections.abc import Iterable
from tensorly.decomposition._nn_cp import make_svd_non_negative

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>

# License: BSD 3 clause


def initialize_tucker(tensor, rank, modes, random_state, init='svd', svd='numpy_svd', non_negative= False):
    """
    Initialize core and factors used in `tucker`.
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.
    
    Parameters
    ----------
    tensor : ndarray
    rank : int
           number of components
    modes : int list
    random_state : {None, int, np.random.RandomState}
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'numpy_svd'
          function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned
    
    Returns
    -------
    core    : ndarray
              initialized core tensor 
    factors : list of factors
    """
    try:
        svd_fun = tl.SVD_FUNS[svd]
    except KeyError:
        message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                svd, tl.get_backend(), tl.SVD_FUNS)
        raise ValueError(message)
    # Initialisation
    if init == 'svd':
        factors = []
        for index, mode in enumerate(modes):
            U, S, V = svd_fun(unfold(tensor, mode), n_eigenvecs=rank[index], random_state=random_state)   
            
            if non_negative is True: 
                U = make_svd_non_negative(tensor, U, S, V, nntype="nndsvd")
            
            factors.append(U[:, :rank[index]])        
        # The initial core approximation is needed here for the masking step
        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)        
        if non_negative is True:
            core = tl.abs(core) 
            
    elif init == 'random':
        rng = tl.check_random_state(random_state)
        core = tl.tensor(rng.random_sample(rank) + 0.01, **tl.context(tensor))  # Check this
        factors = [tl.tensor(rng.random_sample(s), **tl.context(tensor)) for s in zip(tl.shape(tensor), rank)]
        if non_negative is True:
            factors = [tl.abs(f) for f in factors]
            core = tl.abs(core) 
    else:
        (core, factors) = init
 
    return core, factors
    

def partial_tucker(tensor, modes, rank=None, n_iter_max=100, init='svd', tol=10e-5,
                   svd='numpy_svd', random_state=None, verbose=False, mask=None):
    """Partial tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition exclusively along the provided modes.

    Parameters
    ----------
    tensor : ndarray
    modes : int list
            list of the modes on which to perform the decomposition
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, or TuckerTensor optional
        if a TuckerTensor is provided, this is used for initialization
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD,
        acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).

    Returns
    -------
    core : ndarray
            core tensor of the Tucker decomposition
    factors : ndarray list
            list of factors of the Tucker decomposition.
            with ``core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes``

    """
    if rank is None:
        message = "No value given for 'rank'. The decomposition will preserve the original size."
        warnings.warn(message, Warning)
        rank = [tl.shape(tensor)[mode] for mode in modes]
    elif isinstance(rank, int):
        message = "Given only one int for 'rank' instead of a list of {} modes. Using this rank for all modes.".format(len(modes))
        warnings.warn(message, Warning)
        rank = tuple(rank for _ in modes)
    else:
        rank = tuple(rank)

    if mask is not None and init == "svd":
        message = "Masking occurs after initialization. Therefore, random initialization is recommended."
        warnings.warn(message, Warning)

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
            eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank[index], random_state=random_state)
            factors.append(eigenvecs)

        # The initial core approximation is needed here for the masking step
        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)
    elif init == 'random':
        rng = tl.check_random_state(random_state)
        # len(rank) == len(modes) but we still want a core dimension for the modes not optimized
        core_shape = list(tl.shape(tensor))
        for (i, e) in enumerate(modes):
            core_shape[e] = rank[i]
        core = tl.tensor(rng.random_sample(core_shape), **tl.context(tensor))
        factors = [tl.tensor(rng.random_sample((tl.shape(tensor)[mode], rank[index])), **tl.context(tensor)) for (index, mode) in enumerate(modes)]
    else: 
        (core, factors) = init

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):
        if mask is not None:
            tensor = tensor*mask + multi_mode_dot(core, factors, modes=modes, transpose=False)*(1-mask)

        for index, mode in enumerate(modes):
            core_approximation = multi_mode_dot(tensor, factors, modes=modes, skip=index, transpose=True)
            eigenvecs, _, _ = svd_fun(unfold(core_approximation, mode), n_eigenvecs=rank[index], random_state=random_state)
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

    return (core, factors)


def tucker(tensor, rank, fixed_factors=None, n_iter_max=100, init='svd',
           svd='numpy_svd', tol=10e-5, random_state=None, mask=None, verbose=False):
    """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition:
        ``tensor = [| core; factors[0], ...factors[-1] |]`` [1]_

    Parameters
    ----------
    tensor : ndarray
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    fixed_factors : int list or None, default is None
        if not None, list of modes for which to keep the factors fixed.
        Only valid if a Tucker tensor is provided as init.
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
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).
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
    if fixed_factors:
        try:
            (core, factors) = init
        except:
            raise ValueError(f'Got fixed_factor={fixed_factors} but no appropriate Tucker tensor was passed for "init".')
        
        fixed_factors = sorted(fixed_factors)
        modes_fixed, factors_fixed = zip(*[(i, f) for (i, f) in enumerate(factors) if i in fixed_factors])
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed)
        modes, factors = zip(*[(i, f) for (i, f) in enumerate(factors) if i not in fixed_factors])
        init = (core, list(factors))

        core, new_factors = partial_tucker(tensor, modes, rank=rank, n_iter_max=n_iter_max, init=init,
                                           svd=svd, tol=tol, random_state=random_state, mask=mask,
                                           verbose=verbose)

        factors = list(new_factors)
        for i, e in enumerate(fixed_factors):
            factors.insert(e, factors_fixed[i])
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed, transpose=True)

        return TuckerTensor((core, factors))

    else:
        modes = list(range(tl.ndim(tensor)))
        # TO-DO validate rank for partial tucker as well
        rank = validate_tucker_rank(tl.shape(tensor), rank=rank)

        core, factors = partial_tucker(tensor, modes, rank=rank, n_iter_max=n_iter_max, init=init,
                                       svd=svd, tol=tol, random_state=random_state, mask=mask,
                                       verbose=verbose)
        return TuckerTensor((core, factors))

def non_negative_tucker(tensor, rank, n_iter_max=10, init='svd', tol=10e-5,
                        random_state=None, verbose=False, return_errors=False,
                        normalize_factors=False):
    """Non-negative Tucker decomposition

        Iterative multiplicative update, see [2]_

    Parameters
    ----------
    tensor : ``ndarray``
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    n_iter_max : int
        maximum number of iteration
    init : {'svd', 'random'}
    random_state : {None, int, np.random.RandomState}
    verbose : int , optional
        level of verbosity
    ranks : None or int list
    size of the core tensor
    normalize_factors : if True, aggregates the core which will contain the norms of the factors.

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
       "Non-negative tucker decomposition",
       IEEE Conference on Computer Vision and Pattern Recognition s(CVPR),
       pp 1-8, 2007
    """
    rank = validate_tucker_rank(tl.shape(tensor), rank=rank)

    epsilon = 10e-12

    # Initialisation
    if init == 'svd':
        core, factors = tucker(tensor, rank)
        nn_factors = [tl.abs(f) for f in factors]
        nn_core = tl.abs(core)
    else:
        rng = tl.check_random_state(random_state)
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
        if normalize_factors:
            nn_core, nn_factors = tucker_normalize((nn_core, nn_factors))
    tensor = TuckerTensor((nn_core, nn_factors))
    if return_errors:
        return tensor, rec_errors
    else:
        return tensor
    
def non_negative_tucker_hals(tensor, rank, n_iter_max=100, init="svd", svd='numpy_svd', tol=1e-8,
                             sparsity_coefficients=None, core_sparsity_coefficient=None,
                             fixed_modes=None, random_state=None,
                             verbose=False, normalize_factors=False, return_errors=False, exact=False, 
                             algorithm='fista'):
    """
    Non-negative Tucker decomposition

    Uses HALS to update each factor columnwise and uses
    fista or active set algorithm to update the core, see [1]_ 
    
    Parameters
    ----------
    tensor : ndarray
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    n_iter_max : int
            maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
        Default: 1e-8
    sparsity_coefficients : array of float (as much as the number of modes)
        The sparsity coefficients are used for each factor
        If set to None, the algorithm is computed without sparsity
        Default: None
    core_sparsity_coefficient : array of float. This coefficient imposes sparsity on core
        when it is updated with fista.
        Default: None
    fixed_modes : array of integers (between 0 and the number of modes)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: None
    verbose : boolean
        Indicates whether the algorithm prints the successive
        reconstruction errors or not
        Default: False
    normalize_factors : if True, aggregates the core which will contain the norms of the factors.
    return_errors : boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False
    exact : If it is True, the HALS nnls subroutines give results with high precision but it needs high computational cost. 
        If it is False, the algorithm gives an approximate solution.
        Default: False
    algorithm : {'fista', 'active_set'}
        Non negative least square solution to update the core. 
        Default: 'fista'
    Returns
    -------
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.

    Notes
    -----
    Tucker decomposes a tensor into a core tensor and list of factors:
    .. math::
        \\begin{equation}
            tensor = [| core; factors[0], ... ,factors[-1] |]
        \\end{equation}

    We solve the following problem for each factor:
    .. math::
        \\begin{equation}
            \\min_{tensor >= 0} ||tensor_[i] - factors[i]\\times core_[i] \\times (\\prod_{i\\neq j}(factors[j]))^T||^2
        \\end{equation}

    If we define two variables such as:
    .. math::
            U = core_[i] \\times (\\prod_{i\\neq j}(factors[j]\\times factors[j]^T)) \\
            M = tensor_[i]

    Gradient of the problem becomes:
    .. math::
        \\begin{equation}
            \\delta = -U^TM + factors[i] \\times U^TU
        \\end{equation}

    In order to calculate UTU and UTM, we define two variables:
    .. math::
        \\begin{equation}
            core_cross = \prod_{i\\neq j}(core_[i] \\times (\\prod_{i\\neq j}(factors[j]\\times factors[j]^T)) \\
            tensor_cross =  \prod_{i\\neq j} tensor_[i] \\times factors_[i]
        \\end{equation}
    Then UTU and UTM becomes:
    .. math::
        \\begin{equation}
            UTU = core_cross_[j] \\times core_[j]^T  \\
            UTM =  (tensor_cross_[j] \\times \\times core_[j]^T)^T
        \\end{equation}
    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    rank = validate_tucker_rank(tl.shape(tensor), rank=rank)
    n_modes = tl.ndim(tensor)
    if sparsity_coefficients is None or not isinstance(sparsity_coefficients, Iterable):
        sparsity_coefficients = [sparsity_coefficients] * n_modes

    if fixed_modes is None:
        fixed_modes = []

    # Avoiding errors
    for fixed_value in fixed_modes:
        sparsity_coefficients[fixed_value] = None
    # Generating the mode update sequence
    modes = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    nn_core, nn_factors = initialize_tucker(tensor, rank, modes, init=init, svd=svd, random_state=random_state, 
                                            non_negative=True)
    # initialisation - declare local variables
    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []

    # Iterate over one step of NTD
    for iteration in range(n_iter_max):
        # One pass of least squares on each updated mode
        for mode in modes:

            # Computing Hadamard of cross-products
            pseudo_inverse = nn_factors.copy()
            for i, factor in enumerate(nn_factors):
                if i != mode:
                    pseudo_inverse[i] = tl.dot(tl.conj(tl.transpose(factor)), factor)
            # UtU
            core_cross = multi_mode_dot(nn_core, pseudo_inverse, skip=mode)
            UtU = tl.dot(unfold(core_cross, mode), tl.transpose(unfold(nn_core, mode)))

            # UtM
            tensor_cross = multi_mode_dot(tensor, nn_factors, skip=mode, transpose=True)
            MtU = tl.dot(unfold(tensor_cross, mode), tl.transpose(unfold(nn_core, mode)))
            UtM = tl.transpose(MtU)

            # Call the hals resolution with nnls, optimizing the current mode
            nn_factor, _, _, _ = hals_nnls(UtM, UtU, tl.transpose(nn_factors[mode]),
                                           n_iter_max=100, sparsity_coefficient=sparsity_coefficients[mode],
                                           exact=exact)
            nn_factors[mode] = tl.transpose(nn_factor)
        # updating core
        if algorithm == 'fista':
            pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
            core_estimation = multi_mode_dot(tensor, nn_factors, transpose=True)
            learning_rate = 1
            
            for MtM in pseudo_inverse:
                learning_rate *= 1 / (tl.partial_svd(MtM)[1][0])
            nn_core = fista(core_estimation, pseudo_inverse, x=nn_core, n_iter_max=n_iter_max,
                            sparsity_coef=core_sparsity_coefficient, lr=learning_rate,)
        if algorithm == 'active_set':
            pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
            core_estimation_vec = tl.base.tensor_to_vec(tl.tenalg.mode_dot(tensor_cross, tl.transpose(nn_factors[modes[-1]]), modes[-1]))
            pseudo_inverse_kr = tl.tenalg.kronecker(pseudo_inverse)
            vectorcore = active_set_nnls(core_estimation_vec, pseudo_inverse_kr, x=nn_core, n_iter_max=n_iter_max)
            nn_core = tl.reshape(vectorcore, tl.shape(nn_core))
        
        # Adding the l1 norm value to the reconstruction error
        sparsity_error = 0
        for index, sparse in enumerate(sparsity_coefficients):
            if sparse:
                sparsity_error += 2 * (sparse * tl.norm(nn_factors[index], order=1))
        # error computation
        rec_error = tl.norm(tensor - tucker_to_tensor((nn_core, nn_factors)), 2) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconstruction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break
        if normalize_factors:
            nn_core, nn_factors = tucker_normalize((nn_core, nn_factors))
    tensor = TuckerTensor((nn_core, nn_factors))
    if return_errors:
        return tensor, rec_errors
    else:
        return tensor


class Tucker(DecompositionMixin):
    """Tucker decomposition via Higher Order Orthogonal Iteration (HOI).

    Decomposes `tensor` into a Tucker decomposition:
    ``tensor = [| core; factors[0], ...factors[-1] |]`` [1]_

    Parameters
    ----------
    tensor : ndarray
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    non_negative : bool, default is False
        if True, uses a non-negative Tucker via iterative multiplicative updates
        otherwise, uses a Higher-Order Orthogonal Iteration.
    fixed_factors : int list or None, default is None
        if not None, list of modes for which to keep the factors fixed.
        Only valid if a Tucker tensor is provided as init.
    n_iter_max : int
                maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        ignore if non_negative is True
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
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
    SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    def __init__(self, rank=None, n_iter_max=100,
                 init='svd', svd='numpy_svd', tol=10e-5, fixed_factors=None,
                 random_state=None, mask=None, verbose=False):
        self.rank = rank
        self.fixed_factors = fixed_factors
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.tol = tol
        self.random_state = random_state
        self.mask = mask
        self.verbose = verbose

    def fit_transform(self, tensor):
        tucker_tensor = tucker(
            tensor,
            rank=self.rank,
            fixed_factors=self.fixed_factors,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            tol=self.tol,
            random_state=self.random_state,
            mask=self.mask,
            verbose=self.verbose,
        )        
        self.decomposition_ = tucker_tensor
        return tucker_tensor

    # def transform(self, tensor):
    #     _, factors = self.decomposition_
    #     return tlg.multi_mode_dot(tensor, factors, transpose=True)

    # def inverse_transform(self, tensor):
    #     _, factors = self.decomposition_
    #     return tlg.multi_mode_dot(tensor, factors)

    def __repr__(self):
        return f'Rank-{self.rank} Tucker decomposition via HOOI.'


class Tucker_NN(DecompositionMixin):
    """Non-Negative Tucker decomposition via iterative multiplicative update.

    Decomposes `tensor` into a Tucker decomposition:
    ``tensor = [| core; factors[0], ...factors[-1] |]`` [1]_

    Parameters
    ----------
    tensor : ndarray
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    non_negative : bool, default is False
        if True, uses a non-negative Tucker via iterative multiplicative updates
        otherwise, uses a Higher-Order Orthogonal Iteration.
    n_iter_max : int
                maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        ignore if non_negative is True
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
    def __init__(self, rank=None, n_iter_max=100,
                 init='svd', svd='numpy_svd', tol=10e-5, 
                 random_state=None, verbose=False, normalize_factors=False):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.normalize_factors = normalize_factors
        self.init = init
        self.svd = svd
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, tensor):
        tucker_tensor, errors = non_negative_tucker(tensor, rank=self.rank,
                            n_iter_max=self.n_iter_max,
                            normalize_factors=self.normalize_factors,
                            init=self.init,
                            tol=self.tol,
                            random_state=self.random_state,
                            verbose=self.verbose,
                            return_errors=True)
        self.decomposition_ = tucker_tensor
        self.errors_ = errors
        return tucker_tensor

    # def transform(self, tensor):
    #     _, factors = self.decomposition_
    #     return tlg.multi_mode_dot(tensor, factors, transpose=True)

    # def inverse_transform(self, tensor):
    #     _, factors = self.decomposition_
    #     return tlg.multi_mode_dot(tensor, factors)

    def __repr__(self):
        return f'Rank-{self.rank} Non-Negative Tucker decomposition via multiplicative updates.'
