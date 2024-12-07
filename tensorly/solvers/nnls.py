import tensorly as tl
from math import sqrt


def hals_nnls(
    UtM,
    UtU,
    V=None,
    n_iter_max=500,
    tol=1e-8,
    sparsity_coefficient=None,
    ridge_coefficient=None,
    nonzero_rows=False,
    exact=False,
    epsilon=0.0,
    callback=None,
):
    """
    Non Negative Least Squares (NNLS)

    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

    This algorithm is a simplified implementation of the accelerated HALS defined in [1]. It features an early stop stopping criterion. It is simplified to ensure reproducibility and expose a simple API to control the number of inner iterations.

    This function is made for being used repetively inside an
    outer-loop alternating algorithm, for instance for computing nonnegative
    matrix Factorization or tensor factorization. To use as a stand-alone solver, set the exact flag to True.

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
    n_iter_max: Positive integer
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
        Default: 0
    callback: callable, optional
        A callable called after each iteration. The supported signature is

            callback(V: tensor, error: float)

        where V is the last estimated nonnegative least squares solution, and error is the squared Euclidean norm of the difference between V at the current iteration k, and V at iteration k-1 (therefore error is not the loss function which is costly to compute).
        Moreover, the algorithm will also terminate if the callback callable returns True.
        Default: None

    Returns
    -------
    V: array
        a r-by-n nonnegative matrix, see notes.

    Notes
    -----
    We solve the following problem

    .. math::

            \\min_{V >= \\epsilon} ||M-UV||_F^2

    The matrix V is updated linewise. The update rule for this resolution is

    .. math::

            \\begin{equation}
                V[k,:]_{(j+1)} = V[k,:]_{(j)} + (UtM[k,:] - UtU[k,:]\\times V_{(j)})/UtU[k,k]
            \\end{equation}

    with j the update iteration index. V is then thresholded to be larger than epsilon.

    This problem can also be defined by adding respectively a sparsity coefficient and a ridge coefficients

    .. math:: \lambda_s, \lambda_r

    enhancing sparsity or smoothness in the solution [2]. In this sparse/ridge version, the update rule becomes

    .. math::

            \\begin{equation}
                V[k,:]_{(j+1)} = V[k,:]_{(j)} + (UtM[k,:] - UtU[k,:]\\times V_{(j)} - \lambda_s)/(UtU[k,k]+2\lambda_r)
            \\end{equation}

    Note that the data fitting is halved but not the ridge penalization.

    References
    ----------
    .. [1] N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
       Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
       Neural Computation 24 (4): 1085-1105, 2012.

    .. [2] J. Eggert, and E. Korner. "Sparse coding and NMF."
       2004 IEEE International Joint Conference on Neural Networks
       (IEEE Cat. No. 04CH37541). Vol. 4. IEEE, 2004.

    """

    rank, _ = tl.shape(UtM)
    if V is None:
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
                num = UtM[k, :] - tl.dot(UtU[k, :], V) + UtU[k, k] * V[k, :]
                den = UtU[k, k]

                if sparsity_coefficient is not None:
                    num -= sparsity_coefficient
                if ridge_coefficient is not None:
                    den += 2 * ridge_coefficient

                newV = tl.clip(num / den, a_min=epsilon)
                rec_error += tl.norm(V - newV) ** 2
                V = tl.index_update(V, tl.index[k, :], newV)

                # Safety procedure, if columns aren't allow to be zero
                if nonzero_rows and tl.all(V[k, :] == 0):
                    V[k, :] = tl.eps(V.dtype) * tl.max(V)
            elif nonzero_rows:
                raise ValueError(
                    "Column " + str(k) + " of U is zero with nonzero condition"
                )

        if callback is not None:
            retVal = callback(V, rec_error)
            if retVal is True:
                print("Received True from callback function. Exiting.")
                break

        if iteration == 0:
            rec_error0 = rec_error
        if rec_error < tol * rec_error0:
            break

    return V


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
    epsilon=1e-8,
):
    """
    Fast Iterative Shrinkage Thresholding Algorithm (FISTA), see [1]_

    Computes an approximate (nonnegative) solution for Ux=M linear system.

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
        stopping criterion for the l1 error decrease relative to the first iteration error
    epsilon : float
        Small constant such that the solution is greater than epsilon instead of zero.
        Required to ensure convergence, avoids division by zero and reset.
        Default: 1e-8

    Returns
    -------
    x : approximate solution such that Ux = M

    Notes
    -----
    We solve the following problem

    .. math::

            \\frac{1}{2} \\|m - Ux \\|_2^2 + \\lambda_1 \\|x\\|_1 + \\lambda_2 \\|x\\|_2^2

    References
    ----------
    .. [1] Beck, A., & Teboulle, M. (2009). A fast iterative
       shrinkage-thresholding algorithm for linear inverse problems.
       SIAM journal on imaging sciences, 2(1), 183-202.
    """
    if sparsity_coef is None:
        sparsity_coef = 0

    if x is None:
        x = tl.zeros(tl.shape(UtM), **tl.context(UtM))
    if lr is None:
        lr = 1 / (tl.truncated_svd(UtU)[1][0] + 2 * ridge_coef)
    # Parameters
    momentum_old = 1.0  # tl.tensor(1.0)
    norm_0 = 0.0
    x_update = tl.copy(x)

    for iteration in range(n_iter_max):
        if isinstance(UtU, list):
            x_gradient = (
                -UtM
                + tl.tenalg.multi_mode_dot(x_update, UtU, transpose=False)
                + sparsity_coef
                + 2 * ridge_coef * x_update
            )
        else:
            x_gradient = (
                -UtM + tl.dot(UtU, x_update) + sparsity_coef + 2 * ridge_coef * x_update
            )

        x_new = x_update - lr * x_gradient
        if non_negative:
            x_new = tl.where(x_new < epsilon, epsilon, x_new)
        momentum = (1 + sqrt(1 + 4 * momentum_old**2)) / 2
        x_update = x_new + ((momentum_old - 1) / momentum) * (x_new - x)
        momentum_old = momentum
        norm = tl.abs(
            tl.sum(x - x_new)
        )  # for tracking loss decrease, l2 has square overflow issues
        x = tl.copy(x_new)
        if iteration == 0:
            norm_0 = norm
        if norm < tol * norm_0:
            break
    return x


def active_set_nnls(Utm, UtU, x=None, n_iter_max=100, tol=10e-8):
    """
    Active set algorithm for non-negative least square solution, see [1]_

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
                    \\min_{x} \\|Ux - m\\|^2
            \\end{equation}

    According to [1], non-negativity-constrained least square estimation problem becomes:

    .. math::

            \\begin{equation}
                    x' = Utm - UtU x
            \\end{equation}

    References
    ----------
    .. [1] Bro, R., & De Jong, S. (1997). A fast non‐negativity‐constrained
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
