import tensorly as tl
import numpy as np
import math
from tensorly.base import unfold
from tensorly.tenalg import khatri_rao

# Author : Jeremy Cohen


def fast_gradient_step(input_tensor, factors,
                       rank, norm_tensor,
                       aux_factors,
                       step = 0.1, alpha = 0.5,
                       qstep = 0,
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
    input_tensor : tensor of arbitrary order.
    input_factors : current estimates for the PARAFAC decomposition of
    input_tensor. The value of input_factor[update_mode] will be updated using
    a least squares update.
    rank : rank of the decomposition.
    norm_tensor : the Frobenius norm of the input tensor
    step : size of the update step for the gradient. If possible, choose 1/L
           where L is the Lipschitz constant.
    fixed_modes : indexes of modes that are not updated
    weights : entry-wise weights for each data point, contained in a tensor.
    constraints : a list containing constraints codes (see class Parafac).

    Returns -------
    out_factors : updated inputs factors
    out_factors_aux : auxilliary factors, necessary for looping
    alpha : updated value of averaging variable
    rec_error : residual error after the ALS steps.

    TODO :
    Handle missing data
    Handle personal projection operator
    """
    # Generating the mode update sequence
    gen = (mode for mode in range(tl.ndim(input_tensor)) if mode not in fixed_modes)

    # Initialization of factors and aux_factors
    old_factors = np.copy(factors)

    # Gradient initialization
    grad = np.copy(factors)

    # alpha, beta variables update
    alpha_new = 1/2*(qstep-alpha**2 + math.sqrt(
        (qstep-alpha**2)**2 + 4*alpha**2))
    beta = alpha*(1-alpha)/(alpha**2+alpha_new)
    alpha = alpha_new

    # Computing the gradient for updated modes:
    for mode in gen:
        # Unfolding
        # TODO: precompute unfoldings
        unfoldY = unfold(input_tensor,mode)

        # Computing Hadamard of cross-products
        cross = tl.tensor(np.ones((rank, rank)), **tl.context(input_tensor))
        for i, factor in enumerate(aux_factors):
            if i != mode:
                cross = cross*tl.dot(tl.transpose(factor),factor)

        # Computing the Khatri Rao product
        krao = khatri_rao(aux_factors,skip_matrix=mode)
        rhs = tl.dot(unfoldY,krao)

        # Compute the gradient for given mode
        grad[mode] = - rhs + tl.dot(aux_factors[mode], cross)

    # Rebuilding the loop
    gen = (mode for mode in range(tl.ndim(input_tensor)) if mode not in fixed_modes)

    # Updates and Extrapolation
    for mode in gen:
        # Gradient step
        factors[mode] = aux_factors[mode] - step*grad[mode]

        # Projection step
        # TODO : proximal operator API, customisable
        # factor[mode] = prox(factor,'proj_method')
        if constraints and constraints[mode] == 'NN':
            factors[mode][factors[mode] < 0] = 0

        # Extrapolation step
        aux_factors[mode] = factors[mode] + beta*(factors[mode] - old_factors[mode])

    # error computation (improved using precomputed quantities)
    rec_error = norm_tensor ** 2 - 2*tl.dot(
        tl.tensor_to_vec(factors[mode]), tl.tensor_to_vec(
            rhs)) + tl.norm(tl.dot(factors[mode], tl.transpose(krao)), 2)**2
    rec_error = rec_error ** (1/2) / norm_tensor

    return rec_error
