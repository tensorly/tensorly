import tensorly as tl


def validate_constraints(
    non_negative=None,
    l1_reg=None,
    l2_reg=None,
    l2_square_reg=None,
    unimodality=None,
    normalize=None,
    simplex=None,
    normalized_sparsity=None,
    soft_sparsity=None,
    smoothness=None,
    monotonicity=None,
    hard_sparsity=None,
    n_const=1,
    order=0,
):
    """
    Validates input constraints for constrained parafac decomposition and returns a constraint and a parameter for
    proximal operator.

    Parameters
    ----------
    non_negative : bool or dictionary
        This constraint is clipping negative values to '0'.
        If it is True, non-negative constraint is applied to all modes.
    l1_reg : float or list or dictionary, optional
        Penalizes the factor with the l1 norm using the input value as regularization parameter.
    l2_reg : float or list or dictionary, optional
        Penalizes the factor with the l2 norm using the input value as regularization parameter.
    l2_square_reg : float or list or dictionary, optional
        Penalizes the factor with the l2 square norm using the input value as regularization parameter.
    unimodality : bool or dictionary, optional
        If it is True, unimodality constraint is applied to all modes.
        Applied to each column seperately.
    normalize : bool or dictionary, optional
        This constraint divides all the values by maximum value of the input array.
        If it is True, normalize constraint is applied to all modes.
    simplex : float or list or dictionary, optional
        Projects on the simplex with the given parameter
        Applied to each column seperately.
    normalized_sparsity : float or list or dictionary, optional
        Normalizes with the norm after hard thresholding
    soft_sparsity : float or list or dictionary, optional
        Impose that the columns of factors have L1 norm bounded by a user-defined threshold.
    smoothness : float or list or dictionary, optional
        Optimizes the factors by solving a banded system
    monotonicity : bool or dictionary, optional
        Projects columns to monotonically decreasing distrbution
        Applied to each column seperately.
        If it is True, monotonicity constraint is applied to all modes.
    hard_sparsity : float or list or dictionary, optional
        Hard thresholding with the given threshold
    n_const : int
        Number of constraints. If it is None, function returns input tensor.
        Default : 1
    order : int
        Specifies which constraint to implement if several constraints are selected as input
        Default : 0
    Returns
    -------
    constraint : string
    parameter : float
    """
    constraints = [None] * n_const
    parameters = [None] * n_const

    constraints_list = [
        non_negative,
        l1_reg,
        l2_reg,
        l2_square_reg,
        unimodality,
        normalize,
        simplex,
        normalized_sparsity,
        soft_sparsity,
        smoothness,
        monotonicity,
        hard_sparsity,
    ]

    constraints_names = [
        "non_negative",
        "l1_reg",
        "l2_reg",
        "l2_square_reg",
        "unimodality",
        "normalize",
        "simplex",
        "normalized_sparsity",
        "soft_sparsity",
        "smoothness",
        "monotonicity",
        "hard_sparsity",
    ]

    # Checking that no mode is constrained twice
    modes_constrained = set()
    for each_constraint in constraints_list:
        if each_constraint:
            if isinstance(each_constraint, dict):
                for mode in each_constraint:
                    if mode in modes_constrained:
                        raise ValueError(
                            "You selected two constraints for the same mode. Consider to check your input"
                        )
                    modes_constrained.add(mode)
            elif isinstance(each_constraint, list):
                for mode in range(len(each_constraint)):
                    if each_constraint[mode]:
                        if mode in modes_constrained:
                            raise ValueError(
                                "You selected two constraints for the same mode. Consider to check your input"
                            )
                        modes_constrained.add(mode)
            else:  # each_constraint is a float or int applied to all modes
                if len(modes_constrained) > 0:
                    raise ValueError(
                        "You selected two constraints for the same mode. Consider to check your input"
                    )
                for i in range(n_const):
                    modes_constrained.add(i)

    def registrer_constraint(list_or_dict_or_float, name_constraint):
        if isinstance(list_or_dict_or_float, dict):
            modes = list(list_or_dict_or_float)
            for i in range(len(modes)):
                constraints[modes[i]] = name_constraint
                parameters[modes[i]] = list_or_dict_or_float[modes[i]]
        else:
            for i in range(len(constraints)):
                constraints[i] = name_constraint
                if isinstance(list_or_dict_or_float, list):
                    parameters[i] = list_or_dict_or_float[i]
                else:
                    parameters[i] = list_or_dict_or_float

    for each_constraint, each_name in zip(constraints_list, constraints_names):
        if each_constraint:
            registrer_constraint(each_constraint, each_name)

    return constraints[order], parameters[order]


def proximal_operator(
    tensor,
    non_negative=None,
    l1_reg=None,
    l2_reg=None,
    l2_square_reg=None,
    unimodality=None,
    normalize=None,
    simplex=None,
    normalized_sparsity=None,
    soft_sparsity=None,
    smoothness=None,
    monotonicity=None,
    hard_sparsity=None,
    n_const=1,
    order=0,
):
    """
    Proximal operator solves a convex optimization problem. Let f be a
    convex proper lower-semicontinuous function, the proximal operator of f is :math:`\\argmin_x(f(x) + 1/2||x - v||_2^2)`.
    This operator can be used to solve constrained optimization problems as a generalization to projections on convex sets.
    Therefore, proximal gradients are used for constrained tensor decomposition problems in the literature.

    Parameters
    ----------
    tensor : ndarray
    non_negative : bool or dictionary
        This constraint is clipping negative values to '0'.
        If it is True, non-negative constraint is applied to all modes.
    l1_reg : float or list or dictionary, optional
        Penalizes the factor with the given regularizer
    l2_reg : float or list or dictionary, optional
        Penalizes the factor with the given regularizer
    l2_square_reg : float or list or dictionary, optional
        Penalizes the factor with the given regularizer
    unimodality : bool or dictionary, optional
        If it is True, unimodality constraint is applied to all modes.
        Applied to each column seperately.
    normalize : bool or dictionary, optional
        This constraint divides all the values by maximum value of the input array.
        If it is True, normalize constraint is applied to all modes.
    simplex : float or list or dictionary, optional
        Projects on the simplex with the given parameter
        Applied to each column seperately.
    normalized_sparsity : float or list or dictionary, optional
        Normalizes with the norm after hard thresholding
    soft_sparsity : float or list or dictionary, optional
        Simplex operator using soft thresholding
    smoothness : float or list or dictionary, optional
        Optimizes the factors by solving a banded system
    monotonicity : bool or dictionary, optional
        Projects columns to monotonically decreasing distrbution
        Applied to each column seperately.
        If it is True, monotonicity constraint is applied to all modes.
    hard_sparsity : float or list or dictionary, optional
        Hard thresholding with the given threshold
    n_const : int
        Number of constraints. If it is None, function returns input tensor.
        Default : 1
    order : int
        Specifies which constraint to implement if several constraints are selected as input
        Default : 0
    Returns
    -------
    tensor : updated tensor according to the selected constraint, which is the solution of the optimization problem above.
             If constraint is None, function returns the same tensor.

    References
    ----------
    .. [1]: Moreau, J. J. (1962). Fonctions convexes duales et points proximaux dans un espace hilbertien.
            Comptes rendus hebdomadaires des séances de l'Académie des sciences, 255, 2897-2899.
    .. [2]: Parikh, N., & Boyd, S. (2014). Proximal algorithms.
            Foundations and Trends in optimization, 1(3), 127-239.
    """
    if n_const is None:
        return tensor
    constraint, parameter = validate_constraints(
        non_negative=non_negative,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        l2_square_reg=l2_square_reg,
        unimodality=unimodality,
        normalize=normalize,
        simplex=simplex,
        normalized_sparsity=normalized_sparsity,
        soft_sparsity=soft_sparsity,
        smoothness=smoothness,
        monotonicity=monotonicity,
        hard_sparsity=hard_sparsity,
        n_const=n_const,
        order=order,
    )
    if constraint is None:
        return tensor
    elif constraint == "non_negative":
        return tl.clip(tensor, 0, tl.max(tensor))
    elif constraint == "l1_reg":
        return soft_thresholding(tensor, parameter)
    elif constraint == "l2_reg":
        return l2_prox(tensor, parameter)
    elif constraint == "l2_square_reg":
        return l2_square_prox(tensor, parameter)
    elif constraint == "unimodality":
        return unimodality_prox(tensor)
    elif constraint == "normalize":
        return tensor / tl.max(tl.abs(tensor))
    elif constraint == "simplex":
        return simplex_prox(tensor, parameter)
    elif constraint == "normalized_sparsity":
        return normalized_sparsity_prox(tensor, parameter)
    elif constraint == "soft_sparsity":
        return soft_sparsity_prox(tensor, parameter)
    elif constraint == "smoothness":
        return smoothness_prox(tensor, parameter)
    elif constraint == "monotonicity":
        return monotonicity_prox(tensor)
    elif constraint == "hard_sparsity":
        return hard_thresholding(tensor, parameter)
    else:
        raise RuntimeError("Invalid constraint name")


def smoothness_prox(tensor, regularizer):
    """Proximal operator for smoothness

    Parameters
    ----------
    tensor : ndarray
    regularizer : float

    Returns
    -------
    ndarray

    """
    diag_matrix = tl.tensor(
        tl.diag(2 * regularizer * tl.ones(tl.shape(tensor)[0]) + 1)
        + tl.diag(-regularizer * tl.ones(tl.shape(tensor)[0] - 1), k=-1)
        + tl.diag(-regularizer * tl.ones(tl.shape(tensor)[0] - 1), k=1),
        **tl.context(tensor)
    )
    return tl.solve(diag_matrix, tensor)


def monotonicity_prox(tensor, decreasing=False):
    """
    This function projects each column of the input array on the set of arrays so that
          x[1] <= x[2] <= ... <= x[n] (decreasing=False)
                        or
          x[1] >= x[2] >= ... >= x[n] (decreasing=True)
    is satisfied columnwise.

    Parameters
    ----------
    tensor : ndarray
    decreasing : If it is True, function returns columnwise
                 monotone decreasing tensor. Otherwise, returned array
                 will be monotone increasing.
                 Default: True

    Returns
    -------
    ndarray
          A tensor of which columns' are monotonic.

    References
    ----------
    .. [1]: G. Chierchia, E. Chouzenoux, P. L. Combettes, and J.-C. Pesquet
            "The Proximity Operator Repository. User's guide"
    """
    if tl.ndim(tensor) == 1:
        tensor = tl.reshape(tensor, [tl.shape(tensor)[0], 1])
    elif tl.ndim(tensor) > 2:
        raise ValueError(
            "Monotonicity prox doesn't support an input which has more than 2 dimensions."
        )
    tensor_mon = tl.copy(tensor)
    if decreasing:
        tensor_mon = tl.flip(tensor_mon, axis=0)
    row, column = tl.shape(tensor_mon)
    cum_sum = tl.cumsum(tensor_mon, axis=0)
    for j in range(column):
        assisted_tensor = tl.zeros([row, row])
        for i in range(row):
            if i == 0:
                assisted_tensor = tl.index_update(
                    assisted_tensor,
                    tl.index[i, i:],
                    cum_sum[i:, j]
                    / tl.tensor(tl.arange(row - i) + 1, **tl.context(tensor)),
                )
            else:
                assisted_tensor = tl.index_update(
                    assisted_tensor,
                    tl.index[i, i:],
                    (cum_sum[i:, j] - cum_sum[i - 1, j])
                    / tl.tensor(tl.arange(row - i) + 1, **tl.context(tensor)),
                )
        tensor_mon = tl.index_update(
            tensor_mon, tl.index[:, j], tl.max(assisted_tensor, axis=0)
        )
        for i in reversed(range(row - 1)):
            if tensor_mon[i, j] > tensor_mon[i + 1, j]:
                tensor_mon = tl.index_update(
                    tensor_mon, tl.index[i, j], tensor_mon[i + 1, j]
                )
    if decreasing:
        tensor_mon = tl.flip(tensor_mon, axis=0)
    return tensor_mon


def unimodality_prox(tensor):
    """
    This function projects each column of the input array on the set of arrays so that
          x[1] <= x[2] <= x[j] >= x[j+1]... >= x[n]
    is satisfied columnwise.

    Parameters
    ----------
    tensor : ndarray

    Returns
    -------
    ndarray
         A tensor of which columns' distribution are unimodal.

    References
    ----------
    .. [1]: Bro, R., & Sidiropoulos, N. D. (1998). Least squares algorithms under
            unimodality and non‐negativity constraints. Journal of Chemometrics:
            A Journal of the Chemometrics Society, 12(4), 223-247.
    """
    if tl.ndim(tensor) == 1:
        tensor = tl.vec_to_tensor(tensor, [tl.shape(tensor)[0], 1])
    elif tl.ndim(tensor) > 2:
        raise ValueError(
            "Unimodality prox doesn't support an input which has more than 2 dimensions."
        )

    tensor_unimodal = tl.copy(tensor)
    monotone_increasing = tl.tensor(monotonicity_prox(tensor), **tl.context(tensor))
    monotone_decreasing = tl.tensor(
        monotonicity_prox(tensor, decreasing=True), **tl.context(tensor)
    )
    # Next line finds mutual peak points
    values = tl.tensor(
        tl.to_numpy((tensor - monotone_decreasing >= 0))
        * tl.to_numpy((tensor - monotone_increasing >= 0)),
        **tl.context(tensor)
    )

    sum_inc = tl.where(
        values == 1,
        tl.cumsum(tl.abs(tensor - monotone_increasing), axis=0),
        tl.tensor(0, **tl.context(tensor)),
    )
    sum_inc = tl.where(
        values == 1,
        sum_inc - tl.abs(tensor - monotone_increasing),
        tl.tensor(0, **tl.context(tensor)),
    )
    sum_dec = tl.where(
        tl.flip(values, axis=0) == 1,
        tl.cumsum(
            tl.abs(tl.flip(tensor, axis=0) - tl.flip(monotone_decreasing, axis=0)),
            axis=0,
        ),
        tl.tensor(0, **tl.context(tensor)),
    )
    sum_dec = tl.where(
        tl.flip(values, axis=0) == 1,
        sum_dec
        - tl.abs(tl.flip(tensor, axis=0) - tl.flip(monotone_decreasing, axis=0)),
        tl.tensor(0, **tl.context(tensor)),
    )

    difference = tl.where(
        values == 1,
        sum_inc + tl.flip(sum_dec, axis=0),
        tl.max(sum_inc + tl.flip(sum_dec, axis=0)),
    )
    min_indice = tl.argmin(tl.tensor(difference), axis=0)
    for i in range(len(min_indice)):
        tensor_unimodal = tl.index_update(
            tensor_unimodal,
            tl.index[: int(min_indice[i]), i],
            monotone_increasing[: int(min_indice[i]), i],
        )
        tensor_unimodal = tl.index_update(
            tensor_unimodal,
            tl.index[int(min_indice[i] + 1) :, i],
            monotone_decreasing[int(min_indice[i] + 1) :, i],
        )
    return tensor_unimodal


def l2_square_prox(tensor, regularizer):
    """
    Proximal operator of (regularizer * ||.||_2^2) (squared l2 norm).

    Parameters
    ----------
    tensor : ndarray
    regularizer : float

    Returns
    -------
    ndarray

    References
    ----------
    .. [1]: Combettes, P. L., & Pesquet, J. C. (2011). Proximal splitting methods in signal processing.
            In Fixed-point algorithms for inverse problems in science and engineering (pp. 185-212).
            Springer, New York, NY.
    """
    return tensor / (1 + 2 * regularizer)


def l2_prox(tensor, regularizer):
    """
    Proximal operator of (regularizer*|| ||_2) (l2 norm).

    This proximal operator is sometimes called block soft thresholding.

    Parameters
    ----------
    tensor : ndarray
    regularizer : float

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
    if norm > regularizer:
        bigger_value = norm
    else:
        bigger_value = regularizer
    return tensor - (tensor * regularizer / bigger_value)


def normalized_sparsity_prox(tensor, threshold):
    """
    Normalized sparsity operator by using hard thresholding.
    The input is projected on the intersection of the unit l2 ball with the set of threshold-sparse vectors
    \\{||x||_2^2=1 and ||x||_0\\leq threshold \\}

    Parameters
    ----------
    tensor : ndarray
    threshold : int
                target sparsity level

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


def soft_sparsity_prox(tensor, threshold):
    """
    Projects the input tensor on the set of tensors with l1 norm smaller than threshold, using Soft Thresholding.

    Parameters
    ----------
    tensor : ndarray
    threshold :

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
           \\lambda: prox_\\lambda (||tensor||_1) \\leq parameter
        \\end{equation}
    """
    return simplex_prox(tl.abs(tensor), threshold) * tl.sign(tensor)


def simplex_prox(tensor, parameter):
    """
    Projects the input tensor on the simplex of radius parameter.

    Parameters
    ----------
    tensor : ndarray
    parameter : float

    Returns
    -------
    ndarray

    References
    ----------
    .. [1]: Held, Michael, Philip Wolfe, and Harlan P. Crowder.
            "Validation of subgradient optimization."
            Mathematical programming 6.1 (1974): 62-88.
    """
    # Making it work for 1-dimensional tensors as well
    if tl.ndim(tensor) > 1:
        row, col = tl.shape(tensor)
    else:
        row = tl.shape(tensor)[0]
        col = 1
        tensor = tl.reshape(tensor, [row, col])
    tensor_sort = tl.flip(tl.sort(tensor, axis=0), axis=0)
    # Broadcasting is used to divide rows by 1,2,3...
    cumsum_min_param_by_k = (tl.cumsum(tensor_sort, axis=0) - parameter) / tl.cumsum(
        tl.ones([row, 1]), axis=0
    )
    # Added -1 to correspond to a Python index
    to_change = tl.sum(tl.where(tensor_sort > cumsum_min_param_by_k, 1, 0), axis=0) - 1
    difference = tl.zeros(col)
    for i in range(col):
        difference = tl.index_update(
            difference, tl.index[i], cumsum_min_param_by_k[to_change[i], i]
        )
    if col > 1:
        return tl.clip(tensor - difference, a_min=0)
    else:
        return tl.tensor_to_vec(tl.clip(tensor - difference, a_min=0))


def hard_thresholding(tensor, number_of_non_zero):
    """
    Proximal operator of the l0 ``norm''
    Keeps greater "number_of_non_zero" elements untouched and sets other elements to zero.

    Parameters
    ----------
    tensor : ndarray
    number_of_non_zero : int

    Returns
    -------
    ndarray
          Thresholded tensor on which the operator has been applied
    """
    tensor_vec = tl.copy(tl.tensor_to_vec(tensor))
    sorted_indices = tl.argsort(
        tl.flip(tl.argsort(tl.abs(tensor_vec), axis=0), axis=0), axis=0
    )
    return tl.reshape(
        tl.where(
            sorted_indices < number_of_non_zero,
            tensor_vec,
            tl.tensor(0, **tl.context(tensor_vec)),
        ),
        tensor.shape,
    )


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
    >>> from tensorly.solvers.proximal import soft_thresholding
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
    return tl.sign(tensor) * tl.clip(tl.abs(tensor) - threshold, a_min=0)


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
    U, s, V = tl.truncated_svd(matrix, n_eigenvecs=min(matrix.shape))
    return tl.dot(U, tl.reshape(soft_thresholding(s, threshold), (-1, 1)) * V)


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
    U, _, V = tl.truncated_svd(matrix, n_eigenvecs=min(matrix.shape))
    return tl.dot(U, V)
