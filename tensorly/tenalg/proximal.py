import tensorly as tl
import numpy as np

# Author: Jean Kossaifi
#         Jeremy Cohen <jeremy.cohen@irisa.fr>
#         Axel Marmoret <axel.marmoret@inria.fr>
#         Caglayan Tuna <caglayantun@gmail.com>

# License: BSD 3 clause


def validate_constraints(non_negative=None, l1_reg=None, l2_reg=None, l2_square_reg=None, unimodality=None,
                         normalize=None, simplex=None, normalized_sparsity=None, soft_sparsity=None, smoothness=None,
                         monotonicity=None, hard_sparsity=None, n_const=1, order=0):
    """
    Validates input constraints for constrained parafac decomposition and returns a constraint and a parameter for
    proximal operator.

    Parameters
    ----------
    non_negative : bool or dictionary
        This constraint is clipping negative values to '0'. If it is True non-negative constraint is applied to all modes.
    l1_reg : float or list or dictionary, optional
    l2_reg : float or list or dictionary, optional
    l2_square_reg : float or list or dictionary, optional
    unimodality : bool or dictionary, optional
        If it is True unimodality constraint is applied to all modes.
    normalize : bool or dictionary, optional
        This constraint divides all the values by maximum value of the input array. If it is True normalize constraint
        is applied to all modes.
    simplex : float or list or dictionary, optional
    normalized_sparsity : float or list or dictionary, optional
    soft_sparsity : float or list or dictionary, optional
    smoothness : float or list or dictionary, optional
    monotonicity : bool or dictionary, optional
    hard_sparsity : float or list or dictionary, optional
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
    if non_negative:
        if isinstance(non_negative, dict):
            modes = list(non_negative)
            for i in range(len(modes)):
                constraints[modes[i]] = 'non_negative'
        else:
            for i in range(len(constraints)):
                constraints[i] = 'non_negative'
    if l1_reg:
        if isinstance(l1_reg, dict):
            modes = list(l1_reg)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'l1_reg'
                parameters[modes[i]] = l1_reg[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'l1_reg'
                if isinstance(l1_reg, list):
                    parameters[i] = l1_reg[i]
                else:
                    parameters[i] = l1_reg
    if l2_reg:
        if isinstance(l2_reg, dict):
            modes = list(l2_reg)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'l2_reg'
                parameters[modes[i]] = l2_reg[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'l2_reg'
                if isinstance(l2_reg, list):
                    parameters[i] = l2_reg[i]
                else:
                    parameters[i] = l2_reg
    if l2_square_reg:
        if isinstance(l2_square_reg, dict):
            modes = list(l2_square_reg)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'l2_square_reg'
                parameters[modes[i]] = l2_square_reg[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'l2_square_reg'
                if isinstance(l2_square_reg, list):
                    parameters[i] = l2_square_reg[i]
                else:
                    parameters[i] = l2_square_reg
    if normalized_sparsity:
        if isinstance(normalized_sparsity, dict):
            modes = list(normalized_sparsity)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'normalized_sparsity'
                parameters[modes[i]] = normalized_sparsity[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'normalized_sparsity'
                if isinstance(normalized_sparsity, list):
                    parameters[i] = normalized_sparsity[i]
                else:
                    parameters[i] = normalized_sparsity
    if soft_sparsity:
        if isinstance(soft_sparsity, dict):
            modes = list(soft_sparsity)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'soft_sparsity'
                parameters[modes[i]] = soft_sparsity[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'soft_sparsity'
                if isinstance(soft_sparsity, list):
                    parameters[i] = soft_sparsity[i]
                else:
                    parameters[i] = soft_sparsity
    if hard_sparsity:
        if isinstance(hard_sparsity, dict):
            modes = list(hard_sparsity)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'hard_sparsity'
                parameters[modes[i]] = hard_sparsity[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'hard_sparsity'
                if isinstance(hard_sparsity, list):
                    parameters[i] = hard_sparsity[i]
                else:
                    parameters[i] = hard_sparsity
    if simplex:
        if isinstance(simplex, dict):
            modes = list(simplex)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'simplex'
                parameters[modes[i]] = simplex[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'simplex'
                if isinstance(simplex, list):
                    parameters[i] = simplex[i]
                else:
                    parameters[i] = simplex
    if smoothness:
        if isinstance(smoothness, dict):
            modes = list(smoothness)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'smoothness'
                parameters[modes[i]] = smoothness[modes[i]]
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'smoothness'
                if isinstance(smoothness, list):
                    parameters[i] = smoothness[i]
                else:
                    parameters[i] = smoothness
    if unimodality:
        if isinstance(unimodality, dict):
            modes = list(unimodality)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'unimodality'
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'unimodality'
    if monotonicity:
        if isinstance(monotonicity, dict):
            modes = list(monotonicity)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'monotonicity'
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'monotonicity'
    if normalize:
        if isinstance(normalize, dict):
            modes = list(normalize)
            for i in range(len(modes)):
                if constraints[modes[i]] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[modes[i]] = 'normalize'
        else:
            for i in range(len(constraints)):
                if constraints[i] is not None:
                    raise ValueError('You selected two constraints for the same mode. Consider to check your input')
                constraints[i] = 'normalize'
    return constraints[order], parameters[order]


def proximal_operator(tensor, non_negative=None, l1_reg=None, l2_reg=None, l2_square_reg=None, unimodality=None,
                      normalize=None, simplex=None, normalized_sparsity=None, soft_sparsity=None,
                      smoothness=None, monotonicity=None, hard_sparsity=None, n_const=1, order=0):
    """
    Proximal operator solves a convex optimization problem. Let f be a
    convex proper lower-semicontinuous function, the proximal operator of f is :math:`\\argmin_x(f(x) + 1/2||x - v||_2^2)`.
    This operator can be used to solve constrained optimization problems as a generalization to projections on convex sets.
    Therefore, proximal gradients are used for constrained tensor decomposition problems in the literature.

    Parameters
    ----------
    tensor : ndarray
    non_negative : bool or dictionary
        This constraint is clipping negative values to '0'. If it is True non-negative constraint is applied to all modes.
    l1_reg : float or list or dictionary, optional
    l2_reg : float or list or dictionary, optional
    l2_square_reg : float or list or dictionary, optional
    unimodality : bool or dictionary, optional
        If it is True unimodality constraint is applied to all modes.
    normalize : bool or dictionary, optional
        This constraint divides all the values by maximum value of the input array. If it is True normalize constraint
        is applied to all modes.
    simplex : float or list or dictionary, optional
    normalized_sparsity : float or list or dictionary, optional
    soft_sparsity : float or list or dictionary, optional
    smoothness : float or list or dictionary, optional
    monotonicity : bool or dictionary, optional
    hard_sparsity : float or list or dictionary, optional
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
    constraint, parameter = validate_constraints(non_negative=non_negative, l1_reg=l1_reg, l2_reg=l2_reg,
                                                 l2_square_reg=l2_square_reg, unimodality=unimodality,
                                                 normalize=normalize, simplex=simplex, normalized_sparsity=normalized_sparsity,
                                                 soft_sparsity=soft_sparsity, smoothness=smoothness,
                                                 monotonicity=monotonicity, hard_sparsity=hard_sparsity,
                                                 n_const=n_const, order=order)
    if constraint is None:
        return tensor
    elif constraint == 'non_negative':
        return tl.clip(tensor, 0, tl.max(tensor))
    elif constraint == 'l1_reg':
        return soft_thresholding(tensor, parameter)
    elif constraint == 'l2_reg':
        return l2_prox(tensor, parameter)
    elif constraint == 'l2_square_reg':
        return l2_square_prox(tensor, parameter)
    elif constraint == 'unimodality':
        return unimodality_prox(tensor)
    elif constraint == 'normalize':
        return tensor / tl.max(tensor)
    elif constraint == 'simplex':
        return simplex_prox(tensor, parameter)
    elif constraint == 'normalized_sparsity':
        return normalized_sparsity_prox(tensor, parameter)
    elif constraint == 'soft_sparsity':
        return soft_sparsity_prox(tensor, parameter)
    elif constraint == 'smoothness':
        return smoothness_prox(tensor, parameter)
    elif constraint == 'monotonicity':
        return monotonicity_prox(tensor)
    elif constraint == 'hard_sparsity':
        return hard_thresholding(tensor, parameter)


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
    diag_matrix = tl.diag(2 * regularizer * tl.ones(tl.shape(tensor)[0]) + 1) + \
                  tl.diag(-regularizer * tl.ones(tl.shape(tensor)[0] - 1), k=-1) + \
                  tl.diag(-regularizer * tl.ones(tl.shape(tensor)[0] - 1), k=1)
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
        raise ValueError("Monotonicity prox doesn't support an input which has more than 2 dimensions.")
    tensor_mon = tl.copy(tensor)
    if decreasing:
        tensor_mon = tl.flip(tensor_mon, axis=0)
    row, column = tl.shape(tensor_mon)
    cum_sum = tl.cumsum(tensor_mon, axis=0)
    for j in range(column):
        assisted_tensor = tl.zeros([row, row])
        for i in range(row):
            if i == 0:
                assisted_tensor = tl.index_update(assisted_tensor, tl.index[i, i:], cum_sum[i:, j]
                                                  / tl.tensor(tl.arange(row - i) + 1, **tl.context(tensor)))
            else:
                assisted_tensor = tl.index_update(assisted_tensor, tl.index[i, i:], (cum_sum[i:, j] - cum_sum[i - 1, j])
                                                  / tl.tensor(tl.arange(row - i) + 1, **tl.context(tensor)))
        tensor_mon = tl.index_update(tensor_mon, tl.index[:, j], tl.max(assisted_tensor, axis=0))
        for i in reversed(range(row - 1)):
            if tensor_mon[i, j] > tensor_mon[i + 1, j]:
                tensor_mon = tl.index_update(tensor_mon, tl.index[i, j], tensor_mon[i + 1, j])
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
        raise ValueError("Unimodality prox doesn't support an input which has more than 2 dimensions.")

    tensor_unimodal = tl.copy(tensor)
    monotone_increasing = tl.tensor(monotonicity_prox(tensor), **tl.context(tensor))
    monotone_decreasing = tl.tensor(monotonicity_prox(tensor, decreasing=True),
                                    **tl.context(tensor))
    # Next line finds mutual peak points
    values = tl.tensor(tl.to_numpy((tensor - monotone_decreasing >= 0)) * tl.to_numpy(
        (tensor - monotone_increasing >= 0)), **tl.context(tensor))

    sum_inc = tl.where(values == 1, tl.cumsum(tl.abs(tensor - monotone_increasing), axis=0), tl.tensor(0, **tl.context(tensor)))
    sum_inc = tl.where(values == 1, sum_inc - tl.abs(tensor - monotone_increasing), tl.tensor(0, **tl.context(tensor)))
    sum_dec = tl.where(tl.flip(values, axis=0) == 1, tl.cumsum(tl.abs(tl.flip(tensor, axis=0) - tl.flip(monotone_decreasing, axis=0)), axis=0), tl.tensor(0, **tl.context(tensor)))
    sum_dec = tl.where(tl.flip(values, axis=0) == 1, sum_dec - tl.abs(tl.flip(tensor, axis=0) - tl.flip(monotone_decreasing, axis=0)), tl.tensor(0, **tl.context(tensor)))

    difference = tl.where(values == 1, sum_inc + tl.flip(sum_dec, axis=0), tl.max(sum_inc + tl.flip(sum_dec, axis=0)))
    min_indice = tl.argmin(tl.tensor(difference), axis=0)

    for i in range(len(min_indice)):
        tensor_unimodal = tl.index_update(tensor_unimodal, tl.index[:int(min_indice[i]), i], monotone_increasing[:int(min_indice[i]), i])
        tensor_unimodal = tl.index_update(tensor_unimodal, tl.index[int(min_indice[i]+1):, i], monotone_decreasing[int(min_indice[i]+1):, i])
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
    return tensor/(1 + 2 * regularizer)


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
    _, col = tl.shape(tensor)
    tensor = tl.clip(tensor, 0, tl.max(tensor))
    tensor_sort = tl.sort(tensor, axis=0, descending=True)

    to_change = tl.sum(tl.where(tensor_sort > (tl.cumsum(tensor_sort, axis=0) - parameter), 1.0, 0.0), axis=0)
    difference = tl.zeros(col)
    for i in range(col):
        if to_change[i] > 0:
            difference = tl.index_update(difference, tl.index[i], tl.cumsum(tensor_sort, axis=0)[int(to_change[i] - 1), i])
    difference = (difference - parameter) / to_change
    return tl.clip(tensor - difference, a_min=0)


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
    sorted_indices = tl.argsort(tl.argsort(tl.abs(tensor_vec), axis=0, descending=True), axis=0)
    return tl.reshape(tl.where(sorted_indices < number_of_non_zero, tensor_vec, tl.tensor(0, **tl.context(tensor_vec))), tensor.shape)


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
            if tl.any(active_set):
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


def admm(UtM, UtU, x, dual_var, n_iter_max=100, n_const=None, order=None, non_negative=None, l1_reg=None,
         l2_reg=None, l2_square_reg=None, unimodality=None, normalize=None,
         simplex=None, normalized_sparsity=None, soft_sparsity=None,
         smoothness=None, monotonicity=None, hard_sparsity=None, tol=1e-4):
    """
    Alternating direction method of multipliers (ADMM) algorithm to minimize a quadratic function under convex constraints.

    Parameters
    ----------
    UtM: ndarray
       Pre-computed product of the transposed of U and M.
    UtU: ndarray
       Pre-computed product of the transposed of U and U.
    x: init
       Default: None
    dual_var : ndarray
               Dual variable to update x
    n_iter_max : int
        Maximum number of iteration
        Default: 100
    n_const : int
        Default : None
    order : int
        Default : None
    non_negative : bool or dictionary
        This constraint is clipping negative values to '0'. If it is True non-negative constraint is applied to all modes.
    l1_reg : float or list or dictionary, optional
    l2_reg : float or list or dictionary, optional
    l2_square : float or list or dictionary, optional
    unimodality : bool or dictionary, optional
        If it is True unimodality constraint is applied to all modes.
    normalize : bool or dictionary, optional
        This constraint divides all the values by maximum value of the input array. If it is True normalize constraint
        is applied to all modes.
    simplex : float or list or dictionary, optional
    normalized_sparsity : float or list or dictionary, optional
    soft_sparsity : float or list or dictionary, optional
    smoothness : float or list or dictionary, optional
    monotonicity : bool or dictionary, optional
    hard_sparsity : float or list or dictionary, optional
    tol : float

    Returns
    -------
    x : Updated ndarray
    x_split : Updated ndarray
    dual_var : Updated ndarray

    Notes
    -----
    ADMM solves the convex optimization problem :math:`\\min_ f(x) + g(z)` where :math: A(x_split) + Bx = c.

    Following updates are iterated to solve the problem::

    .. math::
        \\begin{equation}
            x_split = argmin_(x_split) f(x_split) + (rho/2)||A(x_split) + Bx - c||_2^2
            x = argmin_x g(x) + (rho/2)||A(x_split) + Bx - c||_2^2
            dual_var = dual_var + (Ax + B(x_split) - c)
        \\end{equation}

    where rho is a constant defined by the user.

    Let us define a least square problem such as :math:`\\||Ux - M||^2 + r(x)`.

    ADMM can be adapted to this least square problem as following::

    .. math::
        \\begin{equation}
            x_split = (UtU + rho\times I)\times(UtM + rho\times(x + dual_var)^T)
            x = argmin r(x) + (rho/2)||x - x_split^T + dual_var||_2^2
            dual_var = dual_var + x - x_split^T
        \\end{equation}
    where r is the regularization operator. Here, x can be updated by using proximity operator
    of :math:`x_split^T - dual_var`.

    References
    ----------
    .. [1] Huang, Kejun, Nicholas D. Sidiropoulos, and Athanasios P. Liavas.
           "A flexible and efficient algorithmic framework for constrained matrix and tensor factorization."
           IEEE Transactions on Signal Processing 64.19 (2016): 5052-5065.
    """
    rho = tl.trace(UtU) / tl.shape(x)[1]
    for iteration in range(n_iter_max):
        x_old = tl.copy(x)
        x_split = tl.solve(tl.transpose(UtU + rho * tl.eye(tl.shape(UtU)[1])),
                           tl.transpose(UtM + rho * (x + dual_var)))
        x = proximal_operator(tl.transpose(x_split) - dual_var, non_negative=non_negative, l1_reg=l1_reg,
                              l2_reg=l2_reg, l2_square_reg=l2_square_reg, unimodality=unimodality, normalize=normalize,
                              simplex=simplex, normalized_sparsity=normalized_sparsity,
                              soft_sparsity=soft_sparsity, smoothness=smoothness, monotonicity=monotonicity,
                              hard_sparsity=hard_sparsity, n_const=n_const, order=order)
        if n_const is None:
            x = tl.transpose(tl.solve(tl.transpose(UtU), tl.transpose(UtM)))
            return x, x_split, dual_var
        dual_var = dual_var + x - tl.transpose(x_split)

        dual_residual = x - tl.transpose(x_split)
        primal_residual = x - x_old

        if tl.norm(dual_residual) < tol * tl.norm(x) and tl.norm(primal_residual) < tol * tl.norm(dual_var):
            break
    return x, x_split, dual_var
