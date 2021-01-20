import tensorly as tl

# Author: Jean Kossaifi

# License: BSD 3 clause



def soft_thresholding(tensor, threshold):
    """Soft-thresholding operator

        sign(tensor) * max[abs(tensor) - threshold, 0]

    Parameters
    ----------
    tensor : ndarray
    threshold : float or ndarray with shape tensor.shape
        * If float the threshold is applied to the whole tensor
        * If ndarray, one threshold is applied per elements, 0 values are ignored

    Returns
    -------
    ndarray
        thresholded tensor on which the operator has been applied

    Examples
    --------
    Basic shrinkage

    >>> import tensorly.backend as T
    >>> from tensorly.tenalg.proximal import soft_thresholding
    >>> tensor = tl.tensor([[1, -2, 1.5], [-4, 3, -0.5]])
    >>> soft_thresholding(tensor, 1.1)
    array([[ 0. , -0.9,  0.4],
           [-2.9,  1.9,  0. ]])


    Example with missing values

    >>> mask = tl.tensor([[0, 0, 1], [1, 0, 1]])
    >>> soft_thresholding(tensor, mask*1.1)
    array([[ 1. , -2. ,  0.4],
           [-2.9,  3. ,  0. ]])

    See also
    --------
    svd_thresholding : SVD-thresholding operator
    """
    return tl.sign(tensor)*tl.clip(tl.abs(tensor) - threshold, a_min=0)


def svd_thresholding(matrix, threshold):
    """Singular value thresholding operator

    Parameters
    ----------
    matrix : ndarray
    threshold : float

    Returns
    -------
    ndarray
        matrix on which the operator has been applied

    See also
    --------
    procrustes : procrustes operator
    """
    U, s, V = tl.partial_svd(matrix, n_eigenvecs=min(matrix.shape))
    return tl.dot(U, tl.reshape(soft_thresholding(s, threshold), (-1, 1))*V)


def procrustes(matrix):
    """Procrustes operator

    Parameters
    ----------
    matrix : ndarray

    Returns
    -------
    ndarray
        matrix on which the Procrustes operator has been applied
        has the same shape as the original tensor


    See also
    --------
    svd_thresholding : SVD-thresholding operator
    """
    U, _, V = tl.partial_svd(matrix, n_eigenvecs=min(matrix.shape))
    return tl.dot(U, V)
def hals_nnls_approx(UtM, UtU, in_V, maxiter=500,delta=10e-8,
                  sparsity_coefficient=None, normalize = False,nonzero=False):

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
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
        Default: 10e-8
    sparsity_coefficient: float or None
        The coefficient controling the sparisty level in the objective function.
        If set to None, the problem is solved unconstrained.
        Default: None
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
    if in_V is None:  # checks if V is empty
        V = tl.solve(UtU, UtM)

        V[V < 0] = 0
        # Scaling
        scale = tl.sum(UtM * V) / tl.sum(
            UtU * tl.dot(V, tl.transpose(V)))
        V = tl.dot(scale, V)
    else:
        V = in_V

    rho = 100000
    eps0 = 0
    cnt = 1
    eps = 1


    while eps >= delta * eps0 and cnt <= 1 + 0.5* rho and cnt <= maxiter:
        nodelta = 0
        for k in range(r):

            if UtU[k, k] != 0:
                if sparsity_coefficient != None: # Modifying the objective function for sparsification

                    deltaV = tl.max([(UtM[k, :] - UtU[k, :] @ V - sparsity_coefficient * tl.ones(n)) / UtU[k, k],
                                        -V[k, :]],axis=0)
                    V[k, :] = V[k, :] + deltaV
                else:  # without sparsity

                    if tl.get_backend() == 'pytorch':
                        import torch
                        deltaV = torch.maximum((UtM[k, :] - tl.dot(UtU[k, :], V)) / UtU[k, k],
                                         -V[k, :])
                    else:
                        deltaV = tl.max([(UtM[k, :] - tl.dot(UtU[k, :], V)) / UtU[k, k],
                                         -V[k, :]], axis=0)
                    if tl.get_backend()=='tensorflow':
                        import tensorflow as tf
                        V=tf.Variable(V,dtype='float')
                        V[k, :].assign(V[k, :] + deltaV)
                    else:
                        V[k, :] = V[k, :] + deltaV

                nodelta = nodelta + tl.dot(deltaV, tl.transpose(deltaV))

                # Safety procedure, if columns aren't allow to be zero
                if nonzero and (V[k, :] == 0).all():
                    V[k, :] = 1e-16 * tl.max(V)

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

        rho_up=tl.shape(V)[0]*tl.shape(V)[1]+tl.shape(V)[1]*r
        rho_down=tl.shape(V)[0]*r+tl.shape(V)[0]
        rho=1+(rho_up/rho_down)
        eps = nodelta
        cnt += 1

    return V, eps, cnt, rho
def hals_nnls_exact(UtM, UtU, in_V, maxiter,delta=10e-12,sparsity_coefficient=None):

    """
    =================================
    Non Negative Least Squares (NNLS)
    =================================

    Computes an exact solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

    The NNLS unconstrained problem, as defined in [1], solve the following problem:

            min_{V >= 0} ||M-UV||_F^2

    The matrix V is updated linewise.

    The update rule of the k-th line of V (V[k,:]) for this resolution is:

            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:] V_(j))/UtU[k,k]

    with j the update iteration.


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
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
        Default: 10e-12
    sparsity_coefficient: float or None
        The coefficient controling the sparisty level in the objective function.
        If set to None, the problem is solved unconstrained.
        Default: None

    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
    eps: float
        number of loops authorized by the error stop criterion
    cnt: integer
        final number of update iteration performed


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
        V = tl.solve(UtU, UtM)

        V[V < 0] = 0
        # Scaling
        scale = tl.sum(UtM * V) / tl.sum(
            UtU * tl.dot(V, tl.transpose(V)))
        V = tl.dot(scale, V)
    else:
        V = in_V.copy()

    eps0 = 0
    cnt = 1
    eps = 1
    delta=10e-12
    while eps >= delta * eps0 and cnt <= maxiter:
        nodelta = 0
        for k in range(r):

            if UtU[k, k] != 0:
                if sparsity_coefficient != None:  # Modifying the objective function for sparsification

                    deltaV = tl.max([(UtM[k, :] - UtU[k, :] @ V - sparsity_coefficient * tl.ones(n)) / UtU[k, k],
                                     -V[k, :]], axis=0)
                    V[k, :] = V[k, :] + deltaV
                else:  # without sparsity
                    deltaV = tl.max([(UtM[k, :] - tl.dot(UtU[k, :], V)) / UtU[k, k],
                                     -V[k, :]], axis=0)

                V[k, :] = V[k, :] + deltaV

                nodelta = nodelta + tl.dot(deltaV, tl.transpose(deltaV))


        if cnt == 1:
            eps0 = nodelta

        eps = nodelta
        cnt += 1

    return V, eps, cnt


