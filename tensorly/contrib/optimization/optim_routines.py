import tensorly as tl
import numpy as np
import time
import math
from tensorly.base import unfold
from tensorly.tenalg import khatri_rao
from tensorly.kruskal_tensor import kruskal_to_tensor

# Author : Jeremy Cohen


def least_squares_nway(tensor, in_factors,
                       rank, norm_tensor, fixed_modes,
                       constraints, acc=None,
                       alpha=1, delta=0):
    """ One pass of Alternating Least squares update along all modes

    Update the factors by solving a least squares problem per mode.

    This function is strictly superior to a least squares solver ran on the
    matricized problems min_X ||Y - AX||_F^2 since A is structured as a
    Kronecker product of other factors.

    Parameters
    ----------
    tensor : ndarray
        Tensor of arbitrary order.
    in_factors : list of array
        Current estimates for the PARAFAC decomposition of
        tensor. The value of factor[update_mode]
        will be updated using a least squares update.
        The values in in_factors are not modified.
    rank : int
        Rank of the decomposition.
    norm_tensor : float
        The Frobenius norm of the input tensor
    fixed_modes : list
        Indexes of modes that are not updated
    constraints : list
        List of constraints applied on each factor. See class Parafac
        for more information
    acc :
    alpha :
    delta :

    Returns -------
    factors : list of array
        Updated inputs factors
    rec_error : float
        residual error after the ALS steps.
    """

    # Copy
    factors = in_factors.copy()

    # Generating the mode update sequence
    gen = (mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes)

    #  for mode in range(tl.ndim(input_tensor)):
    for mode in gen:

        # Unfolding
        # TODO : precompute unfoldings
        unfoldY = unfold(tensor, mode)

        # Set timer for acceleration in HALSacc
        tic = time.time()

        # Computing Hadamard of cross-products
        cross = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
        for i, factor in enumerate(factors):
            if i != mode:
                cross = cross*tl.dot(tl.transpose(factor),factor)

        # Computing the Khatri Rao product
        krao = khatri_rao(factors,skip_matrix=mode)
        rhs = tl.dot(unfoldY,krao)

        # End timer for acceleration in HALSacc
        timer = time.time() - tic

        # Verify constraints type for choosing the least squares solver
        if constraints and constraints[mode] == 'NN':
            # Homemade nonnegative least squares solver
            if acc == 'acc':
                factors[mode] = tl.transpose(nnlsHALSacc(tl.transpose(rhs),
                    cross,tl.transpose(factors[mode]),maxiter=100,
                    atime=timer, alpha=alpha, delta=delta)[0])
            else:
                factors[mode] =  tl.transpose(nnlsHALS(tl.transpose(rhs),
                    cross,tl.transpose(factors[mode]),maxiter=1)[0])
        else:
            # Update using backend linear system solver
            # factors[mode] = tl.transpose(
            #        tl.solve(tl.transpose(cross),tl.transpose(rhs)))
            factors[mode] = tl.transpose(
                    tl.solve(cross,tl.transpose(rhs)))

    # error computation (improved using precomputed quantities)
    rec_error = norm_tensor ** 2 - 2*tl.dot(
            tl.tensor_to_vec(factors[mode]),tl.tensor_to_vec(
                rhs)) + tl.norm(tl.dot(factors[mode],tl.transpose(krao)),2)**2
    rec_error = rec_error ** (1/2) / norm_tensor

    # outputs
    return factors, rec_error


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
    Handle missing data
    Handle personal projection operator
    """
    # Generating the mode update sequence
    gen = (mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes)

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
        unfoldY = unfold(tensor,mode)

        # Computing Hadamard of cross-products
        cross = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
        for i, factor in enumerate(aux_factors):
            if i != mode:
                cross = cross*tl.dot(tl.transpose(factor),factor)

        # Computing the Khatri Rao product
        krao = khatri_rao(aux_factors,skip_matrix=mode)
        rhs = tl.dot(unfoldY,krao)

        # Compute the gradient for given mode
        grad[mode] = - rhs + tl.dot(aux_factors[mode], cross)

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
    rec_error = norm_tensor ** 2 - 2*tl.dot(
        tl.tensor_to_vec(factors[mode]), tl.tensor_to_vec(
            rhs)) + tl.norm(tl.dot(factors[mode], tl.transpose(krao)), 2)**2
    rec_error = rec_error ** (1/2) / norm_tensor

    return factors, rec_error


def multiplicative_update_step(tensor, factors,
                               rank, norm_tensor, fixed_modes,
                               epsilon):
    """One step of non-negative CP decomposition

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

    # Generating the mode update sequence
    gen = (mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes)

    for mode in gen:
        # khatri_rao(factors).tl.dot(khatri_rao(factors))
        # simplifies to multiplications
        sub_indices = [i for i in range(tl.ndim(tensor)) if i != mode]
        for i, e in enumerate(sub_indices):
            if i:
                accum = accum*tl.dot(tl.transpose(factors[e]), factors[e])
            else:
                accum = tl.dot(tl.transpose(factors[e]), factors[e])

        numerator = tl.dot(unfold(tensor, mode), khatri_rao(factors, skip_matrix=mode))
        numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
        denominator = tl.dot(factors[mode], accum)
        denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
        factors[mode] = factors[mode] * numerator / denominator

    # TODO : use precomputed quantities
    rec_error = tl.norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor

    # outputs
    return factors, rec_error


# Author : Jeremy Cohen, from Nicolas Gillis code for HALS on Matlab
def nnlsHALS(UtM, UtU, in_V, maxiter=500):
    """ Computes an approximate solution of the following nonnegative least
     squares problem (NNLS)

               min_{V >= 0} ||M-UV||_F^2

     with an exact block-coordinate descent scheme. M is m by n, U is m by r.

     See N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
     Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
     Neural Computation 24 (4): 1085-1105, 2012.


     ****** Input ******
       UtM  : r-by-n matrix
       UtU  : r-by-r matrix
       in_V  : r-by-n initialization matrix
            default: one non-zero entry per column corresponding to the
            clostest column of U of the corresponding column of M
       maxiter: upper bound on the number of iterations (default=500).

       *Remark. M, U and V are not required to be nonnegative.

     ****** Output ******
       V  : an r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
       cnt : number of iterations
    """
    r, n = tl.shape(UtM)

    if not in_V.all():  # checks if V is empty
        V = tl.solve(UtU, UtM)  # Least squares
        V[V < 0] = 0
        # Scaling
        scale = tl.sum(UtM * V)/tl.sum(
            UtU * tl.dot(V, tl.transpose(V)))
        V = tl.dot(scale, V)
    else:
        V = in_V.copy()

    cnt = 1
    while cnt <= maxiter:
        for k in range(r):
            # Update
            # deltaV = tl.maximum((UtM[k,:]-tl.dot(UtU[k,:], V)) / UtU[k,k],-V[k,:])
            deltaV = np.maximum((UtM[k,:]-tl.dot(UtU[k,:], V)) / UtU[k,k],-V[k,:])
            V[k,:] = V[k,:] + deltaV
            # Safety procedure
            if not V[k,:].any():
                V[k,:] = 1e-16*tl.max(V)
        cnt = cnt+1
    return V, cnt


# Accelerated version, using gross-tier stopping criterion
def nnlsHALSacc(UtM, UtU, in_V, maxiter=500, atime=None, alpha=1, delta=0):
    """ Computes an approximate solution of the following nonnegative least
     squares problem (NNLS)

               min_{V >= 0} ||M-UV||_F^2

     with an exact block-coordinate descent scheme. M is m by n, U is m by r.

     See N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
     Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
     Neural Computation 24 (4): 1085-1105, 2012.

     This accelerated function is made for being used repetively inside an
     outer-loop alternating algorithm, for instance for computing nonnegative
     matrix Factorization or tensor factorization.

     It features two accelerations: an early stop stopping criterion, and a
     complexity averaging between precomputations and loops, so as to use large
     precomputations several times.

     ****** Input ******
       UtM  : r-by-n matrix
       UtU  : r-by-r matrix
       in_V : r-by-n initialization matrix (mutable)
            default: one non-zero entry per column corresponding to the
            clostest column of U of the corresponding column of M
       maxiter: upper bound on the number of iterations (default=500).
       atime : time taken to do the precomputations UtU and UtM
       alpha : positive float
            ratio between outer computations and inner loops. Typically set to
            0.5 or 1.
       delta : float in [0,1]
            early stop criterion, while err_k > delta*err_0. Set small for
            almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
            of a PARAFAC computation.

       *Remark. M, U and V are not required to be nonnegative.

     ****** Output ******
       V  : an r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
       err: final approximation error
       it : number of iterations
    """

    r, n = tl.shape(UtM)
    if not in_V.all():  # checks if V is empty
        V = tl.solve(UtU, UtM)  # Least squares
        V[V < 0] = 0
        # Scaling
        scale = tl.sum(UtM * V)/tl.sum(
            UtU * tl.dot(V, tl.transpose(V)))
        V = tl.dot(scale, V)
    else:
        V = in_V.copy()

    rho = 100000
    eps0 = 0
    cnt = 1
    eps = 1

    # Start timer
    tic = time.time()
    while eps >= delta * eps0 and cnt <= 1+alpha*rho and cnt <= maxiter:
        nodelta = 0
        for k in range(r):
            # Update
            # deltaV = tl.maximum((UtM[k,:]-tl.dot(UtU[k,:], V)) / UtU[k,k],-V[k,:])
            deltaV = np.maximum((UtM[k,:]-tl.dot(UtU[k,:], V)) / UtU[k,k],-V[k,:])
            # There is a bug in last version of tensorly, maximum is not
            # suported anymore
            V[k,:] = V[k,:] + deltaV
            nodelta = nodelta + tl.dot(deltaV, tl.transpose(deltaV))
            # Safety procedure
            if not V[k,:].any():
                V[k,:] = 1e-16*tl.max(V)
        if cnt == 1:
            eps0 = nodelta
            # End timer for one iteration
            btime = time.time() - tic
            if atime:  # atime is provided
                # Number of loops authorized
                rho = atime/btime
        eps = nodelta
        cnt = cnt+1

    return V, eps, cnt, rho
