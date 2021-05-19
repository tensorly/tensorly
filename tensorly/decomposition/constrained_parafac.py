import numpy as np
import warnings
from tensorly.decomposition._cp import error_calc, initialize_cp
import tensorly as tl
from tensorly.random import random_cp
from tensorly.base import unfold
from tensorly.cp_tensor import (CPTensor, cp_norm,
                                unfolding_dot_khatri_rao,
                                validate_cp_rank)
from proximal import ADMM, proximal_operator

# Author: Jean Kossaifi
#         Jeremy Cohen <jeremy.cohen@irisa.fr>
#         Caglayan Tuna <caglayantun@gmail.com>

# License: BSD 3 clause

def initialize_constrained_parafac(tensor, rank, constraints, reg_par, prox_par, init, svd, random_state=None):
    r"""Initialize factors used in `parafac`.

    Parameters
    ----------

    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices with uniform distribution using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor. If init is a previously initialized `cp tensor`, all
    the weights are pulled in the last factor and then the weights are set to "1" for the output tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    constraints : string
    constraint_parameters : float
    random_state : {None, int, np.random.RandomState}
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    n_modes = tl.ndim(tensor)
    rng = tl.check_random_state(random_state)
    if constraints is None or len(constraints) != n_modes:
        constraints = [constraints] * n_modes
    if reg_par is None or len(reg_par) != n_modes:
        reg_par = [reg_par] * n_modes
    if prox_par is None or len(prox_par) != n_modes:
        prox_par = [prox_par] * n_modes
    
    if init == 'random':
        weights, factors = random_cp(tl.shape(tensor), rank, normalise_factors=False, **tl.context(tensor))

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                      svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, S, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank)

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, tl.shape(S)[0])
                U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

            if tensor.shape[mode] < rank:
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)

            factors.append(U[:, :rank])

    elif isinstance(init, (tuple, list, CPTensor)):
        try:
            weights, factors = CPTensor(init)

            if tl.all(weights == 1):
                weights, factors = CPTensor((None, factors))
            else:
                weights_avg = tl.prod(weights) ** (1.0 / tl.shape(weights)[0])
                for i in range(len(factors)):
                    factors[i] = factors[i] * weights_avg
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a CPTensor instance'
            )
    else:
        raise ValueError('Initialization method "{}" not recognized'.format(init))
    
    for i in range(n_modes):
            factors[i] = proximal_operator(factors[i], constraint=constraints[i], reg_par=reg_par[i], prox_par=prox_par[i])
    kt = CPTensor((None, factors))
    return kt


def constrained_parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd',
                        tol_rel=1e-8, tol_abs=1e-6, random_state=None,
                        verbose=0, return_errors=False,
                        constraints=None,
                        reg_par=None,
                        prox_par=None,
                        cvg_criterion='abs_rec_error',
                        fixed_modes=None):
    """CANDECOMP/PARAFAC decomposition via alternating optimization of 
    alternating direction method of multipliers (AO-ADMM):
    
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random'}, optional
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol_rel : float, optional
        (Default: 1e-8) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol_rel`.
    tol_abs : float, optional
        (Default: 1e-6) Absolute reconstruction error tolerance for factor update during
        ADMM optimization.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    constraints : string, optional
        If there is only one constraint, this constraint is applied to all modes. Besides, any constraint can be defined
        for each mode by creating a list of constraints. List of possible constraints are as follows:
        {nonnegative, sparse_l1, l2, unimodality, normalize, simplex, normalized_sparsity, soft_sparsity, smoothness, monotonicity}
    reg_par : float list, optional
        If there is only one parameter, this parameter will be used for each contraint.
        Depending on the selected constraint, a parameter should be added.
    prox_par : float list, optional
        If there is only one parameter, this parameter will be used for each contraint.
        Depending on the selected constraint, a parameter should be added.
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None.
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    
    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications", SIAM
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    .. [2] Huang, Kejun, Nicholas D. Sidiropoulos, and Athanasios P. Liavas. 
           "A flexible and efficient algorithmic framework for constrained matrix and tensor factorization." IEEE
           Transactions on Signal Processing 64.19 (2016): 5052-5065.
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)
    n_modes = tl.ndim(tensor)
    if constraints is None or len(constraints) != n_modes:
        constraints = [constraints] * n_modes
    if reg_par is None or len(reg_par) != n_modes:
        reg_par = [reg_par] * n_modes
    if prox_par is None or len(prox_par) != n_modes:
        prox_par = [prox_par] * n_modes

    weights, factors = initialize_constrained_parafac(tensor, rank, constraints=constraints,
                                                      reg_par=reg_par,
                                                      prox_par=prox_par,
                                                      init=init, svd=svd,
                                                      random_state=random_state)

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    #Id = tl.eye(rank, **tl.context(tensor)) * l2_reg

    if fixed_modes is None:
        fixed_modes = []

    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn('You asked for fixing the last mode, which is not supported.\n '
                      'The last mode will not be fixed. Consider using tl.moveaxis()')
        fixed_modes.remove(tl.ndim(tensor) - 1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]
    #ADMM inits
    dual_var = []
    factors_t = []
    constraint_error_all = []
    for i in range(len(factors)):
        dual_var.append(tl.zeros(tl.shape(factors[i])))
        factors_t.append(tl.transpose(tl.zeros(tl.shape(factors[i]))))

    for iteration in range(n_iter_max):
        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor), factor)
            #pseudo_inverse += Id

            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

            factor, factors_t[mode], dual_var[mode] = ADMM(mttkrp, pseudo_inverse, factors[mode], dual_var[mode], n_iter_max=n_iter_max, constraint=constraints[mode],
                                                       reg_par=reg_par[mode], prox_par=prox_par[mode], tol=tol_abs)
            factors[mode] = factor

        factors_norm = cp_norm((weights, factors))
        iprod = tl.sum(tl.sum(mttkrp * factor, axis=0) * weights)
        rec_error = tl.sqrt(tl.abs(norm_tensor**2 + factors_norm**2 - 2 * iprod)) / norm_tensor
        rec_errors.append(rec_error)
        constraint_error = tl.zeros(len(modes_list))
        for mode in modes_list:
            constraint_error[mode] = tl.norm(factors[mode] - tl.transpose(factors_t[mode])) / tl.norm(factors[mode])
        constraint_error_all.append(tl.sum(constraint_error))
        if tol_abs:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration,
                                                                                                            rec_error,
                                                                                                            rec_error_decrease))
                    
                if constraint_error_all[-1] < tol_rel:
                    break
                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol_rel
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol_rel
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
                    break

            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    cp_tensor = CPTensor((weights, factors))

    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor

