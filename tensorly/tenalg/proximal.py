import tensorly as tl

# Author: Jean Kossaifi
#         Jeremy Cohen <jeremy.cohen@irisa.fr>
#         Axel Marmoret <axel.marmoret@inria.fr>
#         Caglayan TUna <caglayantun@gmail.com>

# License: BSD 3 clause


def proximal_operator(H, constraint, reg_par=None, prox_par=None):
    """
    Proximal operator solves a convex optimization problem. Let f be a
    convex function, proximal operator of f is :math:`\\argmin(f(x) + 1/2||x - v||_2^2)`.
    This operator can be used to solve constrained optimization problems. Therefore, proximal gradients are used
    for constrained tensor decomposition problems in the literature.

    Parameters
    ----------
    H : ndarray
    constraint : string
             Constraint options : nonnegative, sparse_l1, l2, unimodality,
                                  normalize, simplex, normalized_sparsity,
                                  soft_sparsity, smoothness, monotonicity
    reg_par : float, optional
              Specifically, defined for sparse_l1 and l2.
             Default : None
    prox_par : float, optional
              Specifically, defined for normalized_sparsity and soft_sparsity.
             Default : None
    Returns
    -------
    H : updated H according to the constraint. If constraints is None, function returns the same H.

    References
    ----------
    .. [1]: Moreau, J. J. (1962). Fonctions convexes duales et points proximaux dans un espace hilbertien.
            Comptes rendus hebdomadaires des séances de l'Académie des sciences, 255, 2897-2899.
    .. [2]: Parikh, N., & Boyd, S. (2014). Proximal algorithms.
            Foundations and Trends in optimization, 1(3), 127-239.
    """
    if reg_par is None:
        reg_par = 1e-3
    if prox_par is None:
        prox_par = 1e-3
    if constraint is None:
        return H
    elif constraint == 'nonnegative':
        return tl.clip(H, 0, tl.max(H))
    elif constraint == 'sparse_l1':
        return soft_thresholding(H, reg_par)
    elif constraint == 'l2':
        return l2_prox(H, reg_par)
    elif constraint == 'unimodality':
        return unimodality(H)
    elif constraint == 'normalize':
        return H / tl.max(H)
    elif constraint == 'simplex':
        return simplex(H)
    elif constraint == 'normalized_sparsity':
        return normalized_sparsity(H, prox_par)
    elif constraint == 'soft_sparsity':
        return soft_sparsity(H, prox_par)
    elif constraint == 'smoothness':
        return smoothness(H)
    elif constraint == 'monotoncity':
        return monotonicity(H)

def smoothness(tensor):
    """

    Parameters
    ----------
    tensor : ndarray

    Returns
    -------

    References
    ----------
    .. [1]: Timmerman, M. E., & Kiers, H. A. (2002). Three-way component analysis with
            smoothness constraints. Computational statistics & data analysis, 40(3), 447-470.

    """
    return tl.copy(tensor)

def monotonicity(tensor, decreasing=True):
    """
    This function projects each column of array onto:
          x[1] <= x[2] <= ... <= x[n] or x[1] => x[2] => ... => x[n]

    Parameters
    ----------
    tensor : ndarray
    decreasing : If it is True, function return monotone decreasing. Otherwise, returned array
                 will be monotone increasing.
               Default: True

    Returns
    -------
    H : Monotonic ndarray

    References
    ----------
    .. [1]: Schenker, C., Cohen, J. E., & Acar, E. (2020). A Flexible Optimization Framework for
            Regularized Matrix-Tensor Factorizations with Linear Couplings.
            IEEE Journal of Selected Topics in Signal Processing.
    """
    if decreasing is True:
        tensor_mon = np.minimum.accumulate(tensor)
    else:
        tensor_mon = np.maximum.accumulate(tensor)
    return tensor_mon


def unimodality(tensor):
    """

    Parameters
    ----------
    tensor :

    Returns
    -------

    References
    ----------
    .. [1]: Bro, R., & Sidiropoulos, N. D. (1998). Least squares algorithms under
            unimodality and non‐negativity constraints. Journal of Chemometrics:
            A Journal of the Chemometrics Society, 12(4), 223-247.
    """
    return tensor


def l2_prox(tensor, parameter):
    """
    Proximal operator of the l2 (Euclidean) norm.

    This proximal is also called as block soft thresholding.

    Parameters
    ----------
    tensor : ndarray
    parameter : float

    Returns
    -------
    ndarray

    Notes
    -----
    .. math::
        \\begin{equation}
            prox_{\\gamma} ||x||_2 = (1 - \\gamma / \\max(|x||_2, \\gamma ))\\times x
        \\end{equation}
    """
    norm = tl.norm(tensor)
    bigger_value = tl.where(norm > parameter, norm, parameter)
    return tensor - (tensor * parameter / bigger_value)


def normalized_sparsity(tensor, threshold):
    """

    Parameters
    ----------
    tensor : ndarray
    threshold

    Returns
    -------
    ndarray

    References
    ----------
    .. [1]: Le Magoarou, L., & Gribonval, R. (2016). Flexible multilayer
            sparse approximations of matrices and applications.
            IEEE Journal of Selected Topics in Signal Processing, 10(4), 688-700.

    Notes
    -----
    .. math::
        \\begin{equation}
            prox_\\threshold (||tensor||_0) / ||prox_(\\threshold ||tensor||_0)||_2
        \\end{equation}
    """
    tensor_hard = hard_thresholding(tensor, threshold)
    return tensor_hard / tl.norm(tensor_hard)


def soft_sparsity(tensor, parameter):
    """

    Parameters
    ----------
    tensor : ndarray
    parameter:

    Returns
    -------
    ndarray

    References
    ----------
    .. [1]: Schenker, C., Cohen, J. E., & Acar, E. (2020). A Flexible Optimization Framework for
            Regularized Matrix-Tensor Factorizations with Linear Couplings.
            IEEE Journal of Selected Topics in Signal Processing.

    Notes
    -----
    .. math::
        \\begin{equation}
           \\lambda: prox_\\lambda (||tensor||_1) = parameter
        \\end{equation}
    """
    total_non_zero = np.count_nonzero(tensor)
    current_l1 = tl.sum(tl.abs(tensor))
    if current_l1 <= parameter:
        return tensor
    elif current_l1 > parameter:
        difference = current_l1 - parameter
        threshold = difference/total_non_zero

    tensor_soft = soft_thresholding(tensor, threshold)
    return tensor_soft


def simplex(tensor):
    """Proximal operator of simplex

    Parameters
    ----------
    tensor: ndarray

    Returns
    -------
    ndarray

    References
    ----------
    .. [1]: Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008, July).
            Efficient projections onto the l 1-ball for learning in high dimensions.
            In Proceedings of the 25th international conference on Machine learning (pp. 272-279).

    """
    _, col = tl.shape(tensor)
    tensor = tl.clip(H, 0, tl.max(tensor))
    tensor_sort = tl.sort(tensor, axis=0, descending=True)
    tensor_cum = np.cumsum(tensor_sort, axis=0)

    j = tl.sum(tensor_sort > (tensor_cum-1) / np.cumsum(tl.ones(tl.shape(tensor_cum)), axis=0), axis=0)
    theta = tl.zeros(col)
    for i in range(col):
        if j[i] > 0:
            theta[i] = tensor_cum[j[i]-1, i]
    theta = (theta - 1)/j
    return tl.clip(tensor, 0, tl.max(tensor - theta))


def hard_thresholding(tensor, threshold):
    """Proximal operator of l_0 norm.

    Parameters
    ----------
    tensor : ndarray
    threshold :

    Returns
    -------
    ndarray
    """
    tensor[tensor < threshold] = 0
    return tensor


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


def hals_nnls(UtM, UtU, V=None, n_iter_max=500, tol=10e-8,
              sparsity_coefficient=None, normalize=False, nonzero_rows=False, exact=False):

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
    nonzero_rows: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False
    exact: If it is True, the algorithm gives a results with high precision but it needs high computational cost.
        If it is False, the algorithm gives an approximate solution
        Default: False

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
    We solve the following problem :math:`\\min_{V >= 0} ||M-UV||_F^2`

    The matrix V is updated linewise. The update rule for this resolution is::

    .. math::
        \\begin{equation}
            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:]\\times V_(j))/UtU[k,k]
        \\end{equation}

    with j the update iteration.

    This problem can also be defined by adding a sparsity coefficient,
    enhancing sparsity in the solution [2]. In this sparse version, the update rule becomes::

    .. math::
        \\begin{equation}
            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:]\\times V_(j) - sparsity_coefficient)/UtU[k,k]
        \\end{equation}

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
        scale = tl.sum(UtM * V) / tl.sum(
                       UtU * tl.dot(V, tl.transpose(V)))
        V = V * scale

    if exact:
        n_iter_max = 50000
        tol = 10e-16

    for iteration in range(n_iter_max):
        rec_error = 0
        rec_error0 = 0
        for k in range(rank):

            if UtU[k, k]:
                if sparsity_coefficient is not None:  # Modifying the function for sparsification

                    deltaV = tl.where((UtM[k, :] - tl.dot(UtU[k, :], V) - sparsity_coefficient) / UtU[k, k] > -V[k, :],
                                      (UtM[k, :] - tl.dot(UtU[k, :], V) - sparsity_coefficient) / UtU[k, k], -V[k, :])
                    V = tl.index_update(V, tl.index[k, :], V[k, :] + deltaV)

                else:  # without sparsity

                    deltaV = tl.where((UtM[k, :] - tl.dot(UtU[k, :], V)) / UtU[k, k] > -V[k, :],
                                      (UtM[k, :] - tl.dot(UtU[k, :], V)) / UtU[k, k], -V[k, :])
                    V = tl.index_update(V, tl.index[k, :], V[k, :] + deltaV)

                rec_error = rec_error + tl.dot(deltaV, tl.transpose(deltaV))

                # Safety procedure, if columns aren't allow to be zero
                if nonzero_rows and tl.all(V[k, :] == 0):
                    V[k, :] = tl.eps(V.dtype) * tl.max(V)

            elif nonzero_rows:
                raise ValueError("Column " + str(k) + " of U is zero with nonzero condition")

            if normalize:
                norm = tl.norm(V[k, :])
                if norm != 0:
                    V[k, :] /= norm
                else:
                    sqrt_n = 1/n_col_M ** (1/2)
                    V[k, :] = [sqrt_n for i in range(n_col_M)]
        if iteration == 1:
            rec_error0 = rec_error

        numerator = tl.shape(V)[0]*tl.shape(V)[1]+tl.shape(V)[1]*rank
        denominator = tl.shape(V)[0]*rank+tl.shape(V)[0]
        complexity_ratio = 1+(numerator/denominator)
        if exact:
            if rec_error < tol * rec_error0:
                break
        else:
            if rec_error < tol * rec_error0 or iteration > 1 + 0.5 * complexity_ratio:
                break

    return V, rec_error, iteration, complexity_ratio
    

def fista(UtM, UtU, x=None, n_iter_max=100, non_negative=True, sparsity_coef=0,
          lr=None, tol=10e-8):
    """
    Fast Iterative Shrinkage Thresholding Algorithm (FISTA)

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
    tol : float
        stopping criterion

    Returns
    -------
    x : approximate solution such that Ux = M

    Notes
    -----
    We solve the following problem :math: `1/2 ||m - Ux ||_2^2 + \\lambda |x|_1`

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
        lr = 1 / (tl.partial_svd(UtU)[1][0])
    # Parameters
    momentum_old = tl.tensor(1.0)
    norm_0 = 0.0
    x_update = tl.copy(x)

    for iteration in range(n_iter_max):
        if isinstance(UtU, list):
            x_gradient = - UtM + tl.tenalg.multi_mode_dot(x_update, UtU, transpose=False) + sparsity_coef
        else:
            x_gradient = - UtM + tl.dot(UtU, x_update) + sparsity_coef

        if non_negative is True:
            x_gradient = tl.where(lr * x_gradient < x_update, x_gradient, x_update/lr)

        x_new = x_update - lr * x_gradient
        momentum = (1 + tl.sqrt(1 + 4 * momentum_old ** 2)) / 2
        x_update = x_new + ((momentum_old - 1) / momentum) * (x_new - x)
        momentum_old = momentum
        x = tl.copy(x_new)
        norm = tl.norm(lr * x_gradient)
        if iteration == 1:
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
    if tl.get_backend() == 'tensorflow':
        raise ValueError(
            "Active set is not supported with the tensorflow backend. Consider using fista method with tensorflow.")

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
            passive_solution = tl.solve(UtU[passive_set, :][:, passive_set], Utm[passive_set])
            indice_list = []
            for i in range(tl.shape(support_vec)[0]):
                if passive_set[i]:
                    indice_list.append(i)
                    support_vec = tl.index_update(support_vec, tl.index[int(i)], passive_solution[len(indice_list) - 1])
                else:
                    support_vec = tl.index_update(support_vec, tl.index[int(i)], 0)
        # Start from zeros if solve is not achieved  
        except:
            x_vec = tl.zeros(tl.shape(UtU)[1])
            support_vec = tl.zeros(tl.shape(x_vec), **tl.context(x_vec))
            passive_set = x_vec > 0
            active_set = x_vec <= 0
            if tl.any(active_set)==True:
                indice = tl.argmax(x_gradient)
                passive_set = tl.index_update(passive_set, tl.index[indice], True)
                active_set = tl.index_update(active_set, tl.index[indice], False)
            passive_solution = tl.solve(UtU[passive_set, :][:, passive_set], Utm[passive_set])
            indice_list = []
            for i in range(tl.shape(support_vec)[0]):
                if passive_set[i]:
                    indice_list.append(i)
                    support_vec = tl.index_update(support_vec, tl.index[int(i)], passive_solution[len(indice_list) - 1])
                else:
                    support_vec = tl.index_update(support_vec, tl.index[int(i)], 0)

        # update support vector if it is necessary
        if tl.min(support_vec[passive_set]) <= 0:
            for i in range(len(passive_set)):
                alpha = tl.min(x_vec[passive_set][support_vec[passive_set] <= 0] / (x_vec[passive_set][support_vec[passive_set] <= 0] - support_vec[passive_set][support_vec[passive_set] <= 0]))
                update = alpha * (support_vec - x_vec)
                x_vec = x_vec + update
                passive_set = x_vec > 0
                active_set = x_vec <= 0
                passive_solution = tl.solve(UtU[passive_set, :][:, passive_set], Utm[passive_set])
                indice_list = []
                for i in range(tl.shape(support_vec)[0]):
                    if passive_set[i]:
                        indice_list.append(i)
                        support_vec = tl.index_update(support_vec, tl.index[int(i)], passive_solution[len(indice_list) - 1])
                    else:
                        support_vec = tl.index_update(support_vec, tl.index[int(i)], 0)

                if tl.any(passive_set)!=True or tl.min(support_vec[passive_set]) > 0:
                    break
        # set x to s
        x_vec = tl.clip(support_vec, 0, tl.max(support_vec))

        # gradient update
        x_gradient = Utm - tl.dot(UtU, x_vec)

        if tl.any(active_set)!=True or tl.max(x_gradient[active_set]) <= tol:
            break

    return x_vec


def ADMM(UtM, pseudo_inverse, x, dual_var, n_iter_max=100, constraint=None, reg_par=None, prox_par=None, tol=1e-4):
    """
    Alternating direction method of multipliers (AO-ADMM) algorithm to solve linear equation.

    Parameters
    ----------
    UtM: ndarray
       Pre-computed product of the transposed of U and M.
    pseudo_inverse: ndarray
       Pre-computed product of the transposed of U and U.
    x: init
       Default: None
    dual_var : ndarray
               Dual variable to update x
    n_iter_max : int
        Maximum number of iteration
        Default: 100
    constraint : string, optional
    constraint_parameter : float, optional
    tol : float

    Returns
    -------
    x : Updated ndarray
    x_tilde :
    dual_var :

    References
    ----------
    .. [1] Huang, Kejun, Nicholas D. Sidiropoulos, and Athanasios P. Liavas.
           "A flexible and efficient algorithmic framework for constrained matrix and tensor factorization." IEEE Transactions on Signal Processing 64.19 (2016): 5052-5065.
    """
    if reg_par is None:
        reg_par = 1
    rho = np.trace(pseudo_inverse) / tl.shape(x)[1]
    for iteration in range(n_iter_max):
        x_old = tl.copy(x)
        x_dual = tl.solve(tl.transpose(pseudo_inverse + rho * tl.eye(tl.shape(pseudo_inverse)[1])),
                          tl.transpose(UtM + rho * (x + dual_var)))
        if constraint is not None:
            x = proximal_operator(tl.transpose(x_dual) - dual_var, constraint=constraint, reg_par=reg_par / rho,
                                  prox_par=prox_par)
        else:
            x = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(UtM)))
            return x, x_dual, dual_var
        dual_var = dual_var + x - tl.transpose(x_dual)

        dual_residual = x - tl.transpose(x_dual)
        primal_residual = x - x_old

        if tl.norm(dual_residual) < tol * tl.norm(x) and tl.norm(primal_residual) < tol * tl.norm(dual_var):
            break
    return x, x_dual, dual_var