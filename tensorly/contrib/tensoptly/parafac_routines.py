import time
import math
import numpy as np
import tensorly as tl
from ...kruskal_tensor import unfolding_dot_khatri_rao
from ...kruskal_tensor import kruskal_norm
from .nnls_routines import hals_nnls_acc

# Author : Jeremy Cohen


def one_step_als(tensor, factors, mode,
                 rank, norm_tensor, compute_error):
    """ One least squares update along a given mode in ALS.

    Parameters
    ----------
    tensor : ndarray
        Tensor of arbitrary order.
    factors : list of array
        Current estimates for the PARAFAC decomposition of
        tensor. The value of factor[update_mode]
        will be updated using a least squares update inplace.
    mode : integer
        Index of the mode to be updated.
    rank : int
        Rank of the decomposition.
    norm_tensor : float
        The Frobenius norm of the input tensor
    compute_error : boolean
        Decides wether the fitting error is computed or not

    Returns -------
    rec_error : float
        residual error after the ALS steps.
    """

    # Computing Hadamard of cross-products
    cross = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
    for i, factor in enumerate(factors):
        if i != mode:
            cross = cross*tl.dot(tl.conj(tl.transpose(factor)), factor)

    # Computing the Khatri Rao product
    mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

    # Update using backend linear system solver
    factors[mode] = tl.transpose(tl.solve(cross,tl.transpose(mttkrp)))

    # error computation (improved using precomputed quantities)
    if compute_error:
        factors_norm = tl.sum(tl.sum(cross*tl.dot(tl.conj(tl.transpose(factors[mode])), factors[mode])))
        rec_error = norm_tensor ** 2 + factors_norm - 2*tl.dot(tl.tensor_to_vec(factors[mode]),tl.tensor_to_vec(mttkrp))
        rec_error = rec_error ** (1/2) / norm_tensor
    else:
        rec_error = None

    # outputs
    return rec_error


def one_step_hals(tensor, factors, mode, rank,
                  norm_tensor, compute_error, constraints,
                  alpha=1, delta=0, nonzero=False):
    """ One Hierarchical Least Squares update for a single mode.

    Parameters
    ----------
    tensor : ndarray
        Tensor of arbitrary order.
    factors : list of array
        Current estimates for the PARAFAC decomposition of
        tensor. The value of factor[update_mode]
        will be updated using a least squares update.
    mode : int
        The updated mode
    rank : int
        Rank of the decomposition.
    norm_tensor : float
        The Frobenius norm of the input tensor
    compute_error : boolean
        Decides if the reconstruction error is computed.
    fixed_modes : list
        Indexes of modes that are not updated
    constraints : list of string
        Constraints applied on factors. See class Parafac
        for more information
    alpha :
    delta :


    Returns -------
    rec_error : float
        residual error after the ALS steps.
    """

    # Set timer for acceleration in HALSacc
    tic = time.time()

    # Computing Hadamard of cross-products
    cross = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
    for i, factor in enumerate(factors):
        if i != mode:
            cross = cross*tl.dot(tl.conj(tl.transpose(factor)), factor)

    # Computing the Khatri Rao product
    mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

    # End timer for acceleration in HALSacc
    timer = time.time() - tic

    # Verify constraints type for choosing the least squares solver
    if constraints and constraints[mode] == 'NN':
        # Homemade nonnegative least squares solver
        factors[mode] = tl.transpose(hals_nnls_acc(tl.transpose(mttkrp),
            cross,tl.transpose(factors[mode]),maxiter=100,
            atime=timer, alpha=alpha, delta=delta)[0])
    else:
        # Same update rule as ALS
        # Todo: catch exception Singular rhs
        factors[mode] = tl.transpose(tl.solve(cross,tl.transpose(mttkrp)))

    # error computation (improved using precomputed quantities)
    if compute_error:
        factors_norm = tl.sum(tl.sum(cross*tl.dot(tl.conj(tl.transpose(factors[mode])), factors[mode])))
        rec_error = norm_tensor ** 2 + factors_norm - 2*tl.dot(tl.tensor_to_vec(factors[mode]),tl.tensor_to_vec(mttkrp))
        rec_error = rec_error ** (1/2) / norm_tensor
    else:
        rec_error = None

    # outputs
    return rec_error


def multiplicative_update_step(tensor, factors, mode,
                               rank, norm_tensor, fixed_modes,
                               epsilon, compute_error):
    """One step of non-negative CP decomposition on a given mode.

        Uses multiplicative updates, see [2]_

    Parameters
    ----------
     tensor : ndarray
        Tensor of arbitrary order.
    factors : list of array
        Current estimates for the PARAFAC decomposition of
        tensor. The value of factor[update_mode]
        will be updated using a least squares update.
    rank : int
        Rank of the decomposition.
    norm_tensor : float
        The Frobenius norm of the input tensor.
    fixed_modes : list
        Indexes of modes that are not updated.
    epsilon : float
        Small value to avoid division by zero.

    Returns -------
    factors : list of array
        Updated inputs factors
    rec_error : float
        residual error after the ALS steps.

    References
    ----------
    .. [2] Amnon Shashua and Tamir Hazan,
       "Non-negative tensor factorization with applications to statistics and computer vision",
       In Proceedings of the International Conference on Machine Learning (ICML),
       pp 792-799, ICML, 2005
    """

    # simplifies to multiplications
    sub_indices = [i for i in range(tl.ndim(tensor)) if i != mode]
    for i, e in enumerate(sub_indices):
        if i:
            accum = accum*tl.dot(tl.transpose(factors[e]), factors[e])
        else:
            accum = tl.dot(tl.transpose(factors[e]), factors[e])

    mttkrp = unfolding_dot_khatri_rao(tensor, (None,factors), mode)
    numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
    denominator = tl.dot(factors[mode], accum)
    denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
    factors[mode] = factors[mode] * numerator / denominator

    # TODO : use precomputed quantities
    if compute_error:
        factors_norm = kruskal_norm((None, factors)) ** 2
        rec_error = norm_tensor ** 2 + factors_norm - 2*tl.dot(tl.tensor_to_vec(factors[mode]),tl.tensor_to_vec(mttkrp))
        rec_error = rec_error ** (1/2) / norm_tensor
    else:
        rec_error = None

    # outputs
    return rec_error


def fast_gradient_step(tensor, factors,
                       rank, norm_tensor,
                       aux_factors,
                       step, alpha=0.5,
                       qstep=0,
                       fixed_modes=[],
                       weights=[], constraints=[]):
    """ One step of fast gradient update along all modes

    Updates the factor using one step of a Fast gradient descent algorithm.
    Constraints are imposed using a projection on the feasible set after the
    gradient update of the auxilliary variables. This is formally very similar
    to a gradient descent with extrapolation but practically faster.

    Reference: Yuri Nesterov, Introductory lectures on convex optimization: A
    basic course , vol. 87, Springer Science & Business Media, 2004.

    Parameters
    ----------
    tensor : ndarray
        Tensor of arbitrary order.
    factors : list of array
        Current estimates for the PARAFAC decomposition of
        tensor. The value of factor[update_mode]
        will be updated using a least squares update.
    rank : int
        Rank of the decomposition.
    norm_tensor : float
        The Frobenius norm of the input tensor
    fixed_modes : list
        Indexes of modes that are not updated
    step : float
        Size of the update step for the gradient. If possible, choose 1/L
        where L is the Lipschitz constant.
    fixed_modes : list, optional
        (Default: []) List of components indexes that are not updates. Returned
        values for these indexes are therefore the initialization values.
    weights : NOT SUPPORTED
    constraints : list of strings, optional
        (Default: []) A list containing constraints codes (see *** TODO).


    Returns -------
    factors : list of array
        Updated inputs factors
    rec_error : float
        residual error after the ALS steps.
        Returns -------
    out_factors : updated inputs factors
    out_factors_aux : auxilliary factors, necessary for looping
    alpha : updated value of averaging variable
    rec_error : residual error after the ALS steps.

    TODO :
    Handle missing data with mask
    Handle personal projection operator
    """
    # Generating the mode update sequence
    gen = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    # Initialization of factors and aux_factors
    old_factors = tl.copy(factors)

    # Gradient initialization
    grad = tl.copy(factors)

    # alpha, beta variables update
    alpha_new = 1/2*(qstep-alpha**2 + math.sqrt(
        (qstep-alpha**2)**2 + 4*alpha**2))
    beta = alpha*(1-alpha)/(alpha**2+alpha_new)
    alpha = alpha_new

    # Computing the gradient for updated modes:
    for mode in gen:

        # Computing Hadamard of cross-products
        cross = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
        for i, factor in enumerate(aux_factors):
            if i != mode:
                cross = cross*tl.dot(tl.transpose(factor),factor)

        # Computing the Khatri Rao product
        mttkrp = unfolding_dot_khatri_rao(tensor, (None,aux_factors), mode)

        # Compute the gradient for given mode
        grad[mode] = - mttkrp + tl.dot(aux_factors[mode], cross)

    # Rebuilding the loop
    gen = (mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes)

    # Updates and Extrapolation
    for mode in gen:
        # Gradient step
        factors[mode] = aux_factors[mode] - step*grad[mode]

        # Projection step
        # TODO : proximal operator API, customisable
        # factor[mode] = prox(factor,'proj_method')
        if constraints and constraints[mode] == 'NN':
            factors[mode][factors[mode] < 0] = 0
            # Safety procedure
            if not factors[mode].any():
                factors[mode] = 1e-16*tl.max(tensor)

        # Extrapolation step
        aux_factors[mode] = factors[mode] + beta*(factors[mode] - old_factors[mode])

    # error computation (improved using precomputed quantities)
    factors_norm = tl.sum(tl.sum(cross*tl.dot(tl.conj(tl.transpose(factors[mode])), factors[mode])))
    rec_error = norm_tensor ** 2 + factors_norm - 2*tl.dot(
        tl.tensor_to_vec(factors[mode]), tl.tensor_to_vec(mttkrp))
    rec_error = rec_error ** (1/2) / norm_tensor

    return rec_error
