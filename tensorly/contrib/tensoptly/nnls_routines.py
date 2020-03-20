"""
Created on Fri Jun  7 16:40:44 2019
TODO: replace np.maximum by tl.maximum, add maximum to backend

@authors: amarmore (Axel Marmoret), cohenjer (Jeremy E. Cohen)
"""

import numpy as np
import tensorly as tl
import time

def hals_nnls_acc(UtM, UtU, in_V, maxiter=500, atime=None, alpha=0.5, delta=0.01,
                  sparsity_coefficient=None, normalize=False, nonzero=False):
## Author : Axel Marmoret, based on Jeremy E. Cohen version's of Nicolas Gillis Matlab's code for HALS
    """
    =================================
    Non Negative Least Squares (NNLS)
    =================================

    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

    The NNLS unconstrained problem, as defined in [1], solve the following problem:

            min_{V >= 0} ||M-UV||_F^2

    The matrix V is updated linewise.

    The update rule of the k-th line of V (V[k,:]) for this resolution is:

            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:] V_(j))/UtU[k,k]

    with j the update iteration.

    This problem can also be defined by adding a sparsity coefficient,
    enhancing sparsity in the solution [2]. The problem thus becomes:

            min_{V >= 0} ||M-UV||_F^2 + 2*sparsity_coefficient*(\sum\limits_{j = 0}^{r}||V[k,:]||_1)

    NB: 2*sp for uniformization in the derivative

    In this sparse version, the update rule for V[k,:] becomes:

            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:] V_(j) - sparsity_coefficient)/UtU[k,k]

    This algorithm is defined in [1], as an accelerated version of the HALS algorithm.

    It features two accelerations: an early stop stopping criterion, and a
    complexity averaging between precomputations and loops, so as to use large
    precomputations several times.

    This function is made for being used repetively inside an
    outer-loop alternating algorithm, for instance for computing nonnegative
    matrix Factorization or tensor factorization.

    Parameters
    ----------
    UtM: r-by-n array
        Pre-computed product of the transposed of U and M, used in the update rule
    UtU: r-by-r array
        Pre-computed product of the transposed of U and U, used in the update rule
    in_V: r-by-n initialization matrix (mutable)
        Initialized V array
        By default, is initialized with one non-zero entry per column
        corresponding to the closest column of U of the corresponding column of M.
    maxiter: Postivie integer
        Upper bound on the number of iterations
        Default: 500
    atime: Positive float
        Time taken to do the precomputations UtU and UtM
        Default: None
    alpha: Positive float
        Ratio between outer computations and inner loops, typically set to 0.5 or 1.
        Default: 0.5
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
        Default: 0.01
    sparsity_coefficient: float or None
        The coefficient controling the sparisty level in the objective function.
        If set to None, the problem is solved unconstrained.
        Default: None
    normalize: boolean
        True in order to normalize each of the k-th line of V after the update
        False not to update them
        Default: False
    nonzero: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False

    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
    eps: float
        number of loops authorized by the error stop criterion
    cnt: integer
        final number of update iteration performed
    rho: float
        number of loops authorized by the time stop criterion

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2] J. Eggert, and E. Korner. "Sparse coding and NMF."
    2004 IEEE International Joint Conference on Neural Networks
    (IEEE Cat. No. 04CH37541). Vol. 4. IEEE, 2004.

    """

    r, n = tl.shape(UtM)
    if not in_V.size:  # checks if V is empty
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

            if UtU[k,k] != 0:

                if sparsity_coefficient != None: # Using the sparsifying objective function
                    deltaV = np.maximum((UtM[k,:] - UtU[k,:]@V - sparsity_coefficient * tl.ones(n)) / UtU[k,k], -V[k,:])
                    V[k,:] = V[k,:] + deltaV

                else:
                    deltaV = np.maximum((UtM[k,:]- UtU[k,:]@V) / UtU[k,k],-V[k,:]) # Element wise maximum -> good idea ?
                    V[k,:] = V[k,:] + deltaV

                nodelta = nodelta + tl.dot(deltaV, tl.transpose(deltaV))

                # Safety procedure, if columns aren't allow to be zero
                if nonzero and (V[k,:] == 0).all() :
                    V[k,:] = 1e-16*tl.max(V)

            elif nonzero:
                raise ValueError("Column " + str(k) + " of U is zero with nonzero condition")

            if normalize:
                norm = tl.norm(V[k,:])
                if norm != 0:
                    V[k,:] /= norm
                else:
                    sqrt_n = 1/n ** (1/2)
                    V[k,:] = [sqrt_n for i in range(n)]

        if cnt == 1:
            eps0 = nodelta
            # End timer for one iteration
            btime = max(time.time() - tic, 10e-7) # Avoid division by 0

            if atime:  # atime is provided
                # Number of loops authorized
                rho = atime/btime
        eps = nodelta
        cnt += 1

    return V, eps, cnt, rho


# NNLS resolution while approaching another matrix
def hals_coupling_nnls_acc(UtM, UtU, in_V, Vtarget, mu,
                           maxiter=500, atime=None, alpha=0.5, delta=0.01,
                           normalize=False, nonzero=False):

    """
    ==========================================================
    Non Negative Least Squares (NNLS) with coupling constraint
    ==========================================================

    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.
    The used NNLS resolution algorithm problem is defined in [1],
    and is an accelerated HALS algorithm.

    It features two accelerations: an early stop stopping criterion, and a
    complexity averaging between precomputations and loops, so as to use large
    precomputations several times.

    This function is made for being used repetively inside an
    outer-loop alternating algorithm, for instance for computing nonnegative
    matrix Factorization or tensor factorization.

    Nonetheless, this version is adapted for coupling the returned matrix
    to a second matrix, called Vtarget.
    The optimization problem is defined for PARAFAC2 in [2] as below:

            min_{V >= 0} ||M-UV||_F^2 + mu * ||V - Vtarget||_F^2

    Parameters
    ----------
    UtM: r-by-n array
        Pre-computed product of the transposed of U and M, used in the update rule
    UtU: r-by-r array
        Pre-computed product of the transposed of U and U, used in the update rule
    in_V: r-by-n initialization matrix (mutable)
        Initialized V array
        By default, is initialized with one non-zero entry per column
        corresponding to the closest column of U of the corresponding column of M.
    Vtarget: array
        The matrix for V to approach
    mu: float
        The weight given to coupling in the objective function
    maxiter: Postivie integer
        Upper bound on the number of iterations
        Default: 500
    atime: Positive float
        Time taken to do the precomputations UtU and UtM
        Default: None
    alpha: Positive float
        Ratio between outer computations and inner loops, typically set to 0.5 or 1.
        Default: 0.5
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
        Default: 0.01
    normalize: boolean
        True in order to normalize each of the k-th line of V after the update
        False not to update them
        Default: False
    nonzero: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False

    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2 + mu * ||V - Vtarget||_F^2
    eps: float
        number of loops authorized by the error stop criterion
    cnt: integer
        final number of update iteration performed
    rho: float
        number of loops authorized by the time stop criterion

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2] J. E. Cohen and R. Bro, Nonnegative PARAFAC2: A Flexible Coupling Approach,
    DOI: 10.1007/978-3-319-93764-9_9

    """

    r, n = tl.shape(UtM)
    if not in_V.size:  # checks if V is empty
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
    while cnt <= maxiter and eps >= delta * eps0 and cnt <= 1+alpha*rho:
        nodelta = 0
        for k in range(r):

            if UtU[k,k] != 0:
                # Update
                deltaV = np.maximum((UtM[k,:]-UtU[k,:]@V + mu*(Vtarget[k,:] - V[k,:])) / (UtU[k,k] + mu),-V[k,:])

                V[k,:] = V[k,:] + deltaV

                # Direct update of V
                #V[k,:] = np.maximum((UtM[k,:]-UtU[k,:]@V + UtU[k,k]*V[k,:] + mu*Vtarget[k,:]) / (UtU[k,k] + mu), 0)

                nodelta = nodelta + tl.dot(deltaV, tl.transpose(deltaV))

                if nonzero and (V[k,:] == 0).all() :
                    # Safety procedure if we don't want a column to be zero
                    V[k,:] = 1e-16*tl.max(V)

            elif nonzero:
                raise ValueError("Column " + str(k) + " is zero with nonzero condition")

            if normalize:
                norm = tl.norm(V[k,:])
                if norm != 0:
                    V[k,:] /= norm
                else:
                    sqrt_n = 1/n ** (1/2)
                    V[k,:] = [sqrt_n for i in range(n)]

        if cnt == 1:
            eps0 = nodelta
            # End timer for one iteration
            btime = max(time.time() - tic, 10e-7) # Avoid division by 0

            if atime:  # atime is provided
                # Number of loops authorized
                rho = atime/btime
        eps = nodelta
        cnt = cnt+1

    return V, eps, cnt, rho
