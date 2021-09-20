import numpy as np
import warnings

import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..random import random_cp
from ..base import unfold
from ..cp_tensor import (cp_to_tensor, CPTensor,
                         unfolding_dot_khatri_rao, cp_norm,
                         cp_normalize, validate_cp_rank)

# Authors: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>
#          Chris Swierczewski <csw@amazon.com>
#          Sam Schneider <samjohnschneider@gmail.com>
#          Aaron Meurer <asmeurer@gmail.com>

# License: BSD 3 clause

def initialize_cp(tensor, rank, init='svd', svd='numpy_svd', random_state=None, normalize_factors=False):
    r"""Initialize factors used in `parafac`.

    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices with uniform distribution using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor. If init is a previously initialized `cp tensor`, all
    the weights are pulled in the last factor and then the weights are set to "1" for the output tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned

    Returns
    -------
    factors : CPTensor
        An initial cp tensor.

    """
    rng = tl.check_random_state(random_state)

    if init == 'random':
        kt = random_cp(tl.shape(tensor), rank, normalise_factors=False, random_state=rng, **tl.context(tensor))

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                      svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, S, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank)

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, tl.shape(S)[0])
                U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

            if tensor.shape[mode] < rank:
                # TODO: this is a hack but it seems to do the job for now
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)

            factors.append(U[:, :rank])

        kt = CPTensor((None, factors))

    elif isinstance(init, (tuple, list, CPTensor)):
        # TODO: Test this
        try:         
            if normalize_factors is True:
                warnings.warn('It is not recommended to initialize a tensor with normalizing. Consider normalizing the tensor before using this function')
            
            kt = CPTensor(init)
            weights, factors = kt
            
            if tl.all(weights == 1):
                kt = CPTensor((None, factors))
            else:
                weights_avg = tl.prod(weights) ** (1.0 / tl.shape(weights)[0])
                for i in range(len(factors)):
                    factors[i] = factors[i] * weights_avg
                kt = CPTensor((None, factors))
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a CPTensor instance'
            )
    else:
        raise ValueError('Initialization method "{}" not recognized'.format(init))

    if normalize_factors:
        kt = cp_normalize(kt)

    return kt


def sparsify_tensor(tensor, card):
    """Zeros out all elements in the `tensor` except `card` elements with maximum absolute values. 
    
    Parameters
    ----------
    tensor : ndarray
    card : int
        Desired number of non-zero elements in the `tensor`
        
    Returns
    -------
    ndarray of shape tensor.shape
    """
    if card >= np.prod(tensor.shape):
        return tensor
    bound = tl.sort(tl.abs(tensor), axis=None)[-card]
    
    return tl.where(tl.abs(tensor) < bound, tl.zeros(tensor.shape, **tl.context(tensor)), tensor)


def error_calc(tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp=None):
    r""" Perform the error calculation. Different forms are used here depending upon 
    the available information. If `mttkrp=None` or masking is being performed, then the
    full tensor must be constructed. Otherwise, the mttkrp is used to reduce the calculation cost.

    Parameters
    ----------
    tensor : tensor
    norm_tensor : float
        The l2 norm of tensor.
    weights : tensor
        The current CP weights
    factors : tensor
        The current CP factors
    sparsity : float or int
        Whether we allow for a sparse component
    mask : bool
        Whether masking is being performed.
    mttkrp : tensor or None
        The mttkrp product, if available.

    Returns
    -------
    unnorml_rec_error : float
        The unnormalized reconstruction error.
    tensor : tensor
        The tensor, in case it has been updated by masking.
    norm_tensor: float
        The tensor norm, in case it has been updated by masking.
    """

    # If we have to update the mask we already have to build the full tensor
    if (mask is not None) or (mttkrp is None):
        low_rank_component = cp_to_tensor((weights, factors))

        # Update the tensor based on the mask
        if mask is not None:
            tensor = tensor * mask + low_rank_component * (1 - mask)
            norm_tensor = tl.norm(tensor, 2)

        if sparsity:
            sparse_component = sparsify_tensor(tensor - low_rank_component, sparsity)
        else:
            sparse_component = 0.0
    
        unnorml_rec_error = tl.norm(tensor - low_rank_component - sparse_component, 2)
    else:
        if sparsity:
            low_rank_component = cp_to_tensor((weights, factors))
            sparse_component = sparsify_tensor(tensor - low_rank_component, sparsity)
        
            unnorml_rec_error = tl.norm(tensor - low_rank_component - sparse_component, 2)
        else:
            # ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>
            factors_norm = cp_norm((weights, factors))

            # mttkrp and factor for the last mode. This is equivalent to the
            # inner product <tensor, factorization>
            iprod = tl.sum(tl.sum(mttkrp * factors[-1], axis=0) * weights)
            unnorml_rec_error = tl.sqrt(tl.abs(norm_tensor**2 + factors_norm**2 - 2 * iprod))

    return unnorml_rec_error, tensor, norm_tensor


def parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd',
            normalize_factors=False, orthogonalise=False,
            tol=1e-8, random_state=None,
            verbose=0, return_errors=False,
            sparsity=None,
            l2_reg=0,  mask=None,
            cvg_criterion='abs_rec_error',
            fixed_modes=None,
            svd_mask_repeats=5,
            linesearch=False):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::

        tensor = [|weights; factors[0], ..., factors[-1] |].

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initalization.
        See `initialize_factors`.
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
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None. 
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    sparsity : float or int
        If `sparsity` is not None, we approximate tensor as a sum of low_rank_component and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). `sparsity` denotes desired fraction or number of non-zero elements in the sparse_component of the `tensor`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    svd_mask_repeats: int
        If using a tensor with masked values, this initializes using SVD multiple times to
        remove the effect of these missing values on the initialization.
    linesearch : bool, default is False
        Whether to perform line search as proposed by Bro [3].

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )

          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise

        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications", SIAM 
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.

    .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values." 
           Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.

    .. [3] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and 
           Applications", PhD., University of Amsterdam, 1998
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)
    
    if orthogonalise and not isinstance(orthogonalise, int):
        orthogonalise = n_iter_max

    if linesearch:
        acc_pow = 2.0 # Extrapolate to the iteration^(1/acc_pow) ahead
        acc_fail = 0 # How many times acceleration have failed
        max_fail = 4 # Increase acc_pow with one after max_fail failure

    weights, factors = initialize_cp(tensor, rank, init=init, svd=svd,
                                 random_state=random_state,
                                 normalize_factors=normalize_factors)

    if mask is not None and init == "svd":
        for _ in range(svd_mask_repeats):
            tensor = tensor * mask + tl.cp_to_tensor((weights, factors), mask=1 - mask)

            weights, factors = initialize_cp(tensor, rank, init=init, svd=svd, random_state=random_state, normalize_factors=normalize_factors)

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    Id = tl.eye(rank, **tl.context(tensor)) * l2_reg
    
    if fixed_modes is None:
        fixed_modes = []
    
    if fixed_modes == list(range(tl.ndim(tensor))): # Check If all modes are fixed
        cp_tensor = CPTensor((weights, factors)) # No need to run optimization algorithm, just return the initialization
        return cp_tensor

    if tl.ndim(tensor)-1 in fixed_modes:
        warnings.warn('You asked for fixing the last mode, which is not supported.\n The last mode will not be fixed. Consider using tl.moveaxis()')
        fixed_modes.remove(tl.ndim(tensor)-1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]
        
    if sparsity:
        sparse_component = tl.zeros_like(tensor)
        if isinstance(sparsity, float):
            sparsity = int(sparsity * np.prod(tensor.shape))
        else:
            sparsity = int(sparsity)

    for iteration in range(n_iter_max):
        if orthogonalise and iteration <= orthogonalise:
            factors = [tl.qr(f)[0] if min(tl.shape(f)) >= rank else f for i, f in enumerate(factors)]

        if linesearch and iteration % 2 == 0:
            factors_last = [tl.copy(f) for f in factors]
            weights_last = tl.copy(weights)

        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor), factor)
            pseudo_inverse += Id

            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

            factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse),
                                  tl.transpose(mttkrp)))

            if normalize_factors:
                scales = tl.norm(factor, 2, axis=0)
                weights = tl.where(scales==0, tl.ones(tl.shape(scales), **tl.context(factor)), scales)
                factor = factor / tl.reshape(weights, (1, -1))

            factors[mode] = factor

        # Will we be performing a line search iteration
        if linesearch and iteration % 2 == 0 and iteration > 5:
            line_iter = True
        else:
            line_iter = False

        # Calculate the current unnormalized error if we need it
        if (tol or return_errors) and line_iter is False:
            unnorml_rec_error, tensor, norm_tensor = error_calc(tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp)
        else:
            if mask is not None:
                tensor = tensor * mask + tl.cp_to_tensor((weights, factors), mask=1 - mask)

        # Start line search if requested.
        if line_iter is True:
            jump = iteration ** (1.0 / acc_pow)

            new_weights = weights_last + (weights - weights_last) * jump
            new_factors = [factors_last[ii] + (factors[ii] - factors_last[ii]) * jump for ii in range(tl.ndim(tensor))]

            new_rec_error, new_tensor, new_norm_tensor = error_calc(tensor, norm_tensor, new_weights, new_factors, sparsity, mask)

            if (new_rec_error / new_norm_tensor) < rec_errors[-1]:
                factors, weights = new_factors, new_weights
                tensor, norm_tensor = new_tensor, new_norm_tensor
                unnorml_rec_error = new_rec_error
                acc_fail = 0

                if verbose:
                    print("Accepted line search jump of {}.".format(jump))
            else:
                unnorml_rec_error, tensor, norm_tensor = error_calc(tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp)
                acc_fail += 1

                if verbose:
                    print("Line search failed for jump of {}.".format(jump))

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        rec_error = unnorml_rec_error / norm_tensor
        rec_errors.append(rec_error)

        if tol:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]
                
                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration, rec_error, rec_error_decrease, unnorml_rec_error))

                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")
                
                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
                    break
                    
            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    cp_tensor = CPTensor((weights, factors))
    
    if sparsity:
        sparse_component = sparsify_tensor(tensor - cp_to_tensor((weights, factors)),
                                           sparsity)
        cp_tensor = (cp_tensor, sparse_component)

    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor
    

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
        rng = tl.check_random_state(random_state)
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
            indices_kr = indices_kr * size + indices

    # Compute the Khatri-Rao product for the chosen indices
    sampled_kr = tl.ones((n_samples, rank), **tl.context(matrices[0]))
    for indices, matrix in zip(indices_list, matrices):
        sampled_kr = sampled_kr * matrix[indices, :]

    if return_sampled_rows:
        return sampled_kr, indices_list, indices_kr
    else:
        return sampled_kr, indices_list


def randomised_parafac(tensor, rank, n_samples, n_iter_max=100, init='random', svd='numpy_svd',
                       tol=10e-9, max_stagnation=20, return_errors=False, random_state=None, verbose=1):
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
    return_errors : bool, default is False
        if True, return a list of all errors
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
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)

    rng = tl.check_random_state(random_state)
    weights, factors = initialize_cp(tensor, rank, init=init, svd=svd, random_state=random_state)
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
            rec_error = tl.norm(tensor - cp_to_tensor((weights, factors)), 2) / norm_tensor
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

    if return_errors:
        return CPTensor((weights, factors)), rec_errors
    else:
        return CPTensor((weights, factors))


class CP(DecompositionMixin):
    """Candecomp-Parafac decomposition via Alternating-Least Square

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::

        tensor = [|weights; factors[0], ..., factors[-1] |].

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
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
        Stopping criterion for ALS, works if `tol` is not None. 
        If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
        If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    sparsity : float or int
        If `sparsity` is not None, we approximate tensor as a sum of low_rank_component and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). `sparsity` denotes desired fraction or number of non-zero elements in the sparse_component of the `tensor`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    svd_mask_repeats: int
        If using a tensor with masked values, this initializes using SVD multiple times to
        remove the effect of these missing values on the initialization.
    linesearch : bool, default is False
        Whether to perform line search as proposed by Bro [3].

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          
          * all ones if normalize_factors is False (default), 
          * weights of the (normalized) factors otherwise

        * factors : List of factors of the CP decomposition element `i ` is of shape
          (tensor.shape[i], rank)
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
           SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.

    .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values." 
           Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.

    .. [3] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and 
           Applications", PhD., University of Amsterdam, 1998
    """
    def __init__(self, rank, n_iter_max=100, init='svd', svd='numpy_svd', normalize_factors=False, orthogonalise=False,
                 tol=1e-8, random_state=None, verbose=0, sparsity=None, l2_reg=0,  mask=None,
                 cvg_criterion='abs_rec_error', fixed_modes=None, svd_mask_repeats=5, linesearch=False):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.normalize_factors = normalize_factors
        self.orthogonalise = orthogonalise
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.sparsity = sparsity
        self.l2_reg = l2_reg
        self.mask = mask
        self.cvg_criterion = cvg_criterion
        self.fixed_modes = fixed_modes
        self.svd_mask_repeats = svd_mask_repeats
        self.linesearch = linesearch

    
    def fit_transform(self, tensor):
        """Decompose an input tensor

        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose

        Returns
        -------
        CPTensor
            decomposed tensor
        """
        cp_tensor, errors = parafac(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            normalize_factors=self.normalize_factors,
            orthogonalise=self.orthogonalise,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            sparsity=self.sparsity,
            l2_reg=self.l2_reg,
            mask=self.mask,
            cvg_criterion=self.cvg_criterion,
            fixed_modes=self.fixed_modes,
            svd_mask_repeats=self.svd_mask_repeats,
            linesearch=self.linesearch,
            return_errors=True,
        )
        self.decomposition_ = cp_tensor 
        self.errors_ = errors
        return self.decomposition_

    def __repr__(self):
        return f'Rank-{self.rank} CP decomposition.'


class RandomizedCP(DecompositionMixin):
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
    def __init__(self, rank, n_samples, n_iter_max=100, init='random', svd='numpy_svd',
                 tol=10e-9, max_stagnation=20, random_state=None, verbose=1):
        self.rank = rank
        self.n_samples = n_samples
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.tol = tol
        self.max_stagnation = max_stagnation
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, tensor):
        self.decomposition_, self.errors_ = randomised_parafac(tensor, rank=self.rank, n_samples=self.n_samples,
                n_iter_max=self.n_iter_max, init=self.init, svd=self.svd, tol=self.tol, return_errors=True,
                max_stagnation=self.max_stagnation, random_state=self.random_state, verbose=self.verbose)
        return self.decomposition_
