import tensorly as tl
import numpy as np

def hals_nnls(
    UtM,
    UtU,
    V=None,
    n_iter_max=500,
    tol=1e-8,
    sparsity_coefficient=None,
    ridge_coefficient=None,
    normalize=False, #todo remove
    nonzero_rows=False,
    exact=False, #todo remove
    epsilon=0,
    atime = None, #todo implement ?
    alpha = None, #todo implement
    return_error=None, #todo implement
    M = None, #todo implement, req if return_error is true
):
    """
    Non Negative Least Squares (NNLS)

    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

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
    V: r-by-n initialization matrix (mutable)
        Initialized V array
        By default, is initialized with one non-zero entry per column
        corresponding to the closest column of U of the corresponding column of M.
    n_iter_max: Postivie integer
        Upper bound on the number of iterations
        Default: 500
    tol : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
        Default: 10e-8
    sparsity_coefficient: float or None
        The coefficient controling the sparisty level in the objective function.
        If set to None, the problem is solved unconstrained.
        Default: None
    ridge_coefficient: float or None
        The coefficient controling the ridge (l2) penalty in the objective function.
        If set to None, no ridge penalty is imposed.
        Default: None
    nonzero_rows: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False
    exact: If it is True, the algorithm gives a results with high precision but it needs high computational cost.
        If it is False, the algorithm gives an approximate solution
        Default: False
    epsilon: float
        Small constant such that V>=epsilon instead of V>=0.
        Required to ensure convergence, avoids division by zero and reset.
        Default: 0 (TODO: should be >0 but 0 matches older tensorly version)

    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
    rec_error: float
        number of loops authorized by the error stop criterion
    iteration: integer
        final number of update iteration performed
    complexity_ratio: float
        number of loops authorized by the stop criterion

    Notes
    -----
    We solve the following problem :math:`\\min_{V >= \epsilon} ||M-UV||_F^2`

    The matrix V is updated linewise. The update rule for this resolution is::

    .. math::
        \\begin{equation}
            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:]\\times V_(j))/UtU[k,k]
        \\end{equation}

    with j the update iteration. V is then thresholded to be larger than epsilon.

    This problem can also be defined by adding a sparsity coefficient,
    enhancing sparsity in the solution [2]. In this sparse version, the update rule becomes::

    .. math::
        \\begin{equation}
            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:]\\times V_(j) - sparsity_coefficient)/UtU[k,k]
        \\end{equation}

    TODO: CAREFUL no 1/2 in Ridge but 1/2 in data fitting...

    References
    ----------
    .. [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
       Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
       Neural Computation 24 (4): 1085-1105, 2012.

    .. [2] J. Eggert, and E. Korner. "Sparse coding and NMF."
       2004 IEEE International Joint Conference on Neural Networks
       (IEEE Cat. No. 04CH37541). Vol. 4. IEEE, 2004.

    """

    rank, n_col_M = tl.shape(UtM)
    if V is None:  # checks if V is empty
        V = tl.solve(UtU, UtM)
        V = tl.clip(V, a_min=0, a_max=None)
        # Scaling
        scale = tl.sum(UtM * V) / tl.sum(UtU * tl.dot(V, tl.transpose(V)))
        V = V * scale

    if exact:
        n_iter_max = 50000
        tol = 1e-16

    for iteration in range(n_iter_max):
        rec_error = 0
        for k in range(rank):
            if UtU[k, k]:
                num = UtM[k, :] - tl.dot(UtU[k, :], V) + UtU[k,k] * V[k,:]
                den = UtU[k,k]
                
                # Modifying the function for sparsification
                if sparsity_coefficient is not None:
                    num -= sparsity_coefficient
                if ridge_coefficient is not None:
                    den += 2 * ridge_coefficient
                    
                newV = tl.clip(num / den, a_min=epsilon)
                rec_error += tl.norm(V-newV)**2
                V = tl.index_update(V, tl.index[k, :], newV)
                
                # Safety procedure, if columns aren't allow to be zero
                if nonzero_rows and tl.all(V[k, :] == 0):
                    V[k, :] = tl.eps(V.dtype) * tl.max(V)
            elif nonzero_rows:
                raise ValueError(
                    "Column " + str(k) + " of U is zero with nonzero condition"
                )

            if normalize:
                norm = tl.norm(V[k, :])
                if norm != 0:
                    V[k, :] /= norm
                else:
                    sqrt_n = 1 / n_col_M ** (1 / 2)
                    V[k, :] = [sqrt_n for i in range(n_col_M)]
        if iteration == 0:
            rec_error0 = rec_error

        numerator = tl.shape(V)[0] * tl.shape(V)[1] + tl.shape(V)[1] * rank
        denominator = tl.shape(V)[0] * rank + tl.shape(V)[0]
        complexity_ratio = 1 + (numerator / denominator)
        if rec_error < tol * rec_error0 or exact*(iteration > 1 + 0.5 * complexity_ratio):
            #print(iteration)
            break
    return V, rec_error, iteration, complexity_ratio

def fista(
    UtM,
    UtU,
    x=None,
    n_iter_max=100,
    non_negative=True,
    sparsity_coef=0,
    ridge_coef=0,
    lr=None,
    tol=1e-8,
    epsilon=1e-8
):
    """
    Fast Iterative Shrinkage Thresholding Algorithm (FISTA)

    Computes an approximate (nonnegative) solution for Ux=M linear system.
    TODO update doc, check update

    Parameters
    ----------
    UtM : ndarray
        Pre-computed product of the transposed of U and M
    UtU : ndarray
        Pre-computed product of the transposed of U and U
    x : init
       Default: None
    n_iter_max : int
        Maximum number of iteration
        Default: 100
    non_negative : bool, default is False
                   if True, result will be non-negative
    lr : float
        learning rate
        Default : None
    sparsity_coef : float or None
    ridge_coef : float or None
    tol : float
        stopping criterion

    Returns
    -------
    x : approximate solution such that Ux = M

    Notes
    -----
    We solve the following problem :math: `1/2 ||m - Ux ||_2^2 + \\lambda_1 |x|_1 + \\lambda_2 \|x\|_2^2`

    Reference
    ----------
    [1] : Beck, A., & Teboulle, M. (2009). A fast iterative
          shrinkage-thresholding algorithm for linear inverse problems.
          SIAM journal on imaging sciences, 2(1), 183-202.
    """
    if sparsity_coef is None:
        sparsity_coef = 0

    if x is None:
        x = tl.zeros(tl.shape(UtM), **tl.context(UtM))
    if lr is None:
        lr = 1 / (tl.truncated_svd(UtU)[1][0]+2*ridge_coef)
    # Parameters
    momentum_old = 1.0 #tl.tensor(1.0)
    norm_0 = 0
    x_update = tl.copy(x)

    for iteration in range(n_iter_max):
        if isinstance(UtU, list):
            x_gradient = (
                -UtM
                + tl.tenalg.multi_mode_dot(x_update, UtU, transpose=False)
                + sparsity_coef + 2*ridge_coef*x_update
            )
        else:
            x_gradient = -UtM + tl.dot(UtU, x_update) + sparsity_coef + 2*ridge_coef*x_update

        x_new = x_update - lr * x_gradient
        if non_negative:
            x_new[x_new<epsilon] = epsilon
        momentum = (1 + tl.sqrt(1 + 4 * momentum_old**2)) / 2
        x_update = x_new + ((momentum_old - 1) / momentum) * (x_new - x)
        momentum_old = momentum
        #norm = tl.norm(x - x_new) # has precision issues?? square overflow weird
        norm = tl.sum(x - x_new) # for tracking loss decrease
        x = tl.copy(x_new)
        if iteration == 0:
            norm_0 = norm
        if norm < tol * norm_0:
            break
    return x

def active_set_nnls(Utm, UtU, x=None, n_iter_max=100, tol=10e-8):
    """
    Active set algorithm for non-negative least square solution.

    Computes an approximate non-negative solution for Ux=m linear system.

    Parameters
    ----------
    Utm : vectorized ndarray
       Pre-computed product of the transposed of U and m
    UtU : ndarray
       Pre-computed Kronecker product of the transposed of U and U
    x : init
       Default: None
    n_iter_max : int
        Maximum number of iteration
        Default: 100
    tol : float
        Early stopping criterion

    Returns
    -------
    x : ndarray

    Notes
    -----
    This function solves following problem:
    .. math::
       \\begin{equation}
            \\min_{x} ||Ux - m||^2
       \\end{equation}

    According to [1], non-negativity-constrained least square estimation problem becomes:
    .. math::
       \\begin{equation}
            x' = (Utm) - (UTU)\\times x
       \\end{equation}

    Reference
    ----------
    [1] : Bro, R., & De Jong, S. (1997). A fast non‐negativity‐constrained
          least squares algorithm. Journal of Chemometrics: A Journal of
          the Chemometrics Society, 11(5), 393-401.
    """
    if tl.get_backend() == "tensorflow":
        raise ValueError(
            "Active set is not supported with the tensorflow backend. Consider using fista method with tensorflow."
        )

    if x is None:
        x_vec = tl.zeros(tl.shape(UtU)[1], **tl.context(UtU))
    else:
        x_vec = tl.base.tensor_to_vec(x)

    x_gradient = Utm - tl.dot(UtU, x_vec)
    passive_set = x_vec > 0
    active_set = x_vec <= 0
    support_vec = tl.zeros(tl.shape(x_vec), **tl.context(x_vec))

    for iteration in range(n_iter_max):
        if iteration > 0 or tl.all(x_vec == 0):
            indice = tl.argmax(x_gradient)
            passive_set = tl.index_update(passive_set, tl.index[indice], True)
            active_set = tl.index_update(active_set, tl.index[indice], False)
        # To avoid singularity error when initial x exists
        try:
            passive_solution = tl.solve(
                UtU[passive_set, :][:, passive_set], Utm[passive_set]
            )
            indice_list = []
            for i in range(tl.shape(support_vec)[0]):
                if passive_set[i]:
                    indice_list.append(i)
                    support_vec = tl.index_update(
                        support_vec,
                        tl.index[int(i)],
                        passive_solution[len(indice_list) - 1],
                    )
                else:
                    support_vec = tl.index_update(support_vec, tl.index[int(i)], 0)
        # Start from zeros if solve is not achieved
        except:
            x_vec = tl.zeros(tl.shape(UtU)[1])
            support_vec = tl.zeros(tl.shape(x_vec), **tl.context(x_vec))
            passive_set = x_vec > 0
            active_set = x_vec <= 0
            if tl.any(active_set):
                indice = tl.argmax(x_gradient)
                passive_set = tl.index_update(passive_set, tl.index[indice], True)
                active_set = tl.index_update(active_set, tl.index[indice], False)
            passive_solution = tl.solve(
                UtU[passive_set, :][:, passive_set], Utm[passive_set]
            )
            indice_list = []
            for i in range(tl.shape(support_vec)[0]):
                if passive_set[i]:
                    indice_list.append(i)
                    support_vec = tl.index_update(
                        support_vec,
                        tl.index[int(i)],
                        passive_solution[len(indice_list) - 1],
                    )
                else:
                    support_vec = tl.index_update(support_vec, tl.index[int(i)], 0)

        # update support vector if it is necessary
        if tl.min(support_vec[passive_set]) <= 0:
            for i in range(len(passive_set)):
                alpha = tl.min(
                    x_vec[passive_set][support_vec[passive_set] <= 0]
                    / (
                        x_vec[passive_set][support_vec[passive_set] <= 0]
                        - support_vec[passive_set][support_vec[passive_set] <= 0]
                    )
                )
                update = alpha * (support_vec - x_vec)
                x_vec = x_vec + update
                passive_set = x_vec > 0
                active_set = x_vec <= 0
                passive_solution = tl.solve(
                    UtU[passive_set, :][:, passive_set], Utm[passive_set]
                )
                indice_list = []
                for i in range(tl.shape(support_vec)[0]):
                    if passive_set[i]:
                        indice_list.append(i)
                        support_vec = tl.index_update(
                            support_vec,
                            tl.index[int(i)],
                            passive_solution[len(indice_list) - 1],
                        )
                    else:
                        support_vec = tl.index_update(support_vec, tl.index[int(i)], 0)

                if tl.any(passive_set) != True or tl.min(support_vec[passive_set]) > 0:
                    break
        # set x to s
        x_vec = tl.clip(support_vec, 0, tl.max(support_vec))

        # gradient update
        x_gradient = Utm - tl.dot(UtU, x_vec)

        if tl.any(active_set) != True or tl.max(x_gradient[active_set]) <= tol:
            break

    return x_vec

