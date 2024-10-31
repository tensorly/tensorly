import tensorly as tl
from tensorly.tenalg.proximal import *


def admm(
    UtM,
    UtU,
    x,
    dual_var,
    n_iter_max=100,
    n_const=None,
    order=None,
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
    tol=1e-4,
):
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
        Number of constraints. If it is None, function solves least square problem without proximity operator
        If ADMM function is used with a constraint apart from constrained parafac decomposition,
        n_const value should be changed to '1'.
        Default : None
    order : int
        Specifies which constraint to implement if several constraints are selected as input
        Default : None
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
    tol : float

    Returns
    -------
    x : Updated ndarray
    x_split : Updated ndarray
    dual_var : Updated ndarray

    Notes
    -----
    ADMM solves the convex optimization problem

    .. math:: \\min_ f(x) + g(z),\\; A(x_{split}) + Bx = c.

    Following updates are iterated to solve the problem

    .. math:: x_{split} = argmin_{x_{split}}~ f(x_{split}) + (\\rho/2)\\|Ax_{split} + Bx - c\\|_2^2
    .. math:: x = argmin_x~ g(x) + (\\rho/2)\\|Ax_{split} + Bx - c\\|_2^2
    .. math:: dual\_var = dual\_var + (Ax + Bx_{split} - c)

    where rho is a constant defined by the user.

    Let us define a least square problem such as :math:`\\|Ux - M\\|^2 + r(x)`.

    ADMM can be adapted to this least square problem as following

    .. math:: x_{split} = (UtU + \\rho\\times I)\\times(UtM + \\rho\\times(x + dual\_var)^T)
    .. math:: x = argmin_{x}~ r(x) + (\\rho/2)\\|x - x_{split}^T + dual\_var\\|_2^2
    .. math:: dual\_var = dual\_var + x - x_{split}^T

    where r is the regularization operator. Here, x can be updated by using proximity operator
    of :math:`x_{split}^T - dual\_var`.

    References
    ----------
    .. [1] Huang, Kejun, Nicholas D. Sidiropoulos, and Athanasios P. Liavas.
       "A flexible and efficient algorithmic framework for constrained matrix and tensor factorization."
       IEEE Transactions on Signal Processing 64.19 (2016): 5052-5065.
    """
    rho = tl.trace(UtU) / tl.shape(x)[1]
    for iteration in range(n_iter_max):
        x_old = tl.copy(x)
        x_split = tl.solve(
            tl.transpose(UtU + rho * tl.eye(tl.shape(UtU)[1])),
            tl.transpose(UtM + rho * (x + dual_var)),
        )
        x = proximal_operator(
            tl.transpose(x_split) - dual_var,
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
        if n_const is None:
            x = tl.transpose(tl.solve(tl.transpose(UtU), tl.transpose(UtM)))
            return x, x_split, dual_var
        dual_var = dual_var + x - tl.transpose(x_split)

        dual_residual = x - tl.transpose(x_split)
        primal_residual = x - x_old

        if tl.norm(dual_residual) < tol * tl.norm(x) and tl.norm(
            primal_residual
        ) < tol * tl.norm(dual_var):
            break
    return x, x_split, dual_var
