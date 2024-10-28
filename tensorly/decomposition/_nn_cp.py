import warnings
import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ._cp import initialize_cp
from ..solvers.nnls import hals_nnls
from ..cp_tensor import (
    CPTensor,
    unfolding_dot_khatri_rao,
    cp_norm,
    cp_normalize,
    validate_cp_rank,
)
from ..tenalg.svd import svd_interface

# Authors: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>
#          Chris Swierczewski <csw@amazon.com>
#          Sam Schneider <samjohnschneider@gmail.com>
#          Aaron Meurer <asmeurer@gmail.com>
#          Aaron Meyer <tensorly@ameyer.me>
#          Jeremy Cohen <jeremy.cohen@irisa.fr>
#          Axel Marmoret <axel.marmoret@inria.fr>
#          Caglayan TUna <caglayantun@gmail.com>

# License: BSD 3 clause


def non_negative_parafac(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
    tol=10e-7,
    random_state=None,
    verbose=0,
    normalize_factors=False,
    return_errors=False,
    mask=None,
    cvg_criterion="abs_rec_error",
    fixed_modes=None,
):
    """
    Non-negative CP decomposition

    Uses multiplicative updates, see [2]_

    Parameters
    ----------
    tensor : ndarray
    rank   : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.

    Returns
    -------
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``

    References
    ----------
    .. [2] Amnon Shashua and Tamir Hazan,
           "Non-negative tensor factorization with applications to statistics and computer vision",
           In Proceedings of the International Conference on Machine Learning (ICML),
           pp 792-799, ICML, 2005
    """
    epsilon = tl.eps(tensor.dtype)
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)

    weights, factors = initialize_cp(
        tensor,
        rank,
        init=init,
        svd=svd,
        non_negative=True,
        mask=mask,
        random_state=random_state,
        normalize_factors=normalize_factors,
    )
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    if fixed_modes is None:
        fixed_modes = []

    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn(
            "You asked for fixing the last mode, which is not supported while tol is fixed.\n The last mode will not be fixed. Consider using tl.moveaxis()"
        )
        fixed_modes.remove(tl.ndim(tensor) - 1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    for iteration in range(n_iter_max):
        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            accum = 1
            # khatri_rao(factors).tl.dot(khatri_rao(factors))
            # simplifies to multiplications
            sub_indices = [i for i in range(len(factors)) if i != mode]
            for i, e in enumerate(sub_indices):
                if i:
                    accum *= tl.dot(tl.transpose(factors[e]), factors[e])
                else:
                    accum = tl.dot(tl.transpose(factors[e]), factors[e])
            accum = tl.reshape(weights, (-1, 1)) * accum * tl.reshape(weights, (1, -1))
            if mask is not None:
                tensor = tensor * mask + tl.cp_to_tensor(
                    (weights, factors), mask=1 - mask
                )

            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)

            numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
            denominator = tl.dot(factors[mode], accum)
            denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
            factor = factors[mode] * numerator / denominator

            factors[mode] = factor
            if normalize_factors and mode != modes_list[-1]:
                weights, factors = cp_normalize((weights, factors))

        if tol:
            # ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>
            factors_norm = cp_norm((weights, factors))

            # mttkrp and factor for the last mode. This is equivalent to the
            # inner product <tensor, factorization>
            iprod = tl.sum(tl.sum(mttkrp * factor, axis=0))
            rec_error = (
                tl.sqrt(tl.abs(norm_tensor**2 + factors_norm**2 - 2 * iprod))
                / norm_tensor
            )
            rec_errors.append(rec_error)
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print(
                        f"iteration {iteration}, reconstraction error: {rec_error}, decrease = {rec_error_decrease}"
                    )

                if cvg_criterion == "abs_rec_error":
                    stop_flag = tl.abs(rec_error_decrease) < tol
                elif cvg_criterion == "rec_error":
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print(f"PARAFAC converged after {iteration} iterations")
                    break
            else:
                if verbose:
                    print(f"reconstruction error={rec_errors[-1]}")
        if normalize_factors:
            weights, factors = cp_normalize((weights, factors))
    cp_tensor = CPTensor((weights, factors))

    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor


def non_negative_parafac_hals(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
    tol=10e-8,
    random_state=None,
    sparsity_coefficients=None,
    fixed_modes=None,
    nn_modes="all",
    exact=False,
    normalize_factors=False,
    verbose=False,
    return_errors=False,
    cvg_criterion="abs_rec_error",
):
    """
    Non-negative CP decomposition via HALS

    Uses Hierarchical ALS (Alternating Least Squares)
    which updates each factor column-wise (one column at a time while keeping all other columns fixed), see [1]_

    Parameters
    ----------
    tensor : ndarray
    rank   : int
            number of components
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
          Default: 1e-8
    random_state : {None, int, np.random.RandomState}
    sparsity_coefficients: array of float (of length the number of modes)
        The sparsity coefficients on each factor.
        If set to None, the algorithm is computed without sparsity
        Default: None,
    fixed_modes: array of integers (between 0 and the number of modes)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: None
    nn_modes: None, 'all' or array of integers (between 0 and the number of modes)
        Used to specify which modes to impose non-negativity constraints on.
        If 'all', then non-negativity is imposed on all modes.
        Default: 'all'
    exact: If it is True, the algorithm gives a results with high precision but it needs high computational cost.
        If it is False, the algorithm gives an approximate solution
        Default: False
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    verbose: boolean
        Indicates whether the algorithm prints the successive
        reconstruction errors or not
        Default: False
    return_errors: boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
        Stopping criterion for ALS, works if `tol` is not None.
        If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
        If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    sparsity : float or int
    random_state : {None, int, np.random.RandomState}

    Returns
    -------
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.

    References
    ----------
    .. [1] N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
           Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
           Neural Computation 24 (4): 1085-1105, 2012.
    """

    weights, factors = initialize_cp(
        tensor,
        rank,
        init=init,
        svd=svd,
        non_negative=True,
        random_state=random_state,
        normalize_factors=normalize_factors,
    )

    norm_tensor = tl.norm(tensor, 2)

    n_modes = tl.ndim(tensor)
    if sparsity_coefficients is None or isinstance(sparsity_coefficients, float):
        sparsity_coefficients = [sparsity_coefficients] * n_modes

    if fixed_modes is None:
        fixed_modes = []

    if nn_modes == "all":
        nn_modes = set(range(n_modes))
    elif nn_modes is None:
        nn_modes = set()

    # Avoiding errors
    for fixed_value in fixed_modes:
        sparsity_coefficients[fixed_value] = None

    for mode in range(n_modes):
        if sparsity_coefficients[mode] is not None:
            warnings.warn("Sparsity coefficient is ignored in unconstrained modes.")
    # Generating the mode update sequence
    modes = [mode for mode in range(n_modes) if mode not in fixed_modes]

    # initialisation - declare local varaibles
    rec_errors = []

    # Iteratation
    for iteration in range(n_iter_max):
        # One pass of least squares on each updated mode
        for mode in modes:
            # Computing Hadamard of cross-products
            pseudo_inverse = tl.ones((rank, rank), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(
                        tl.transpose(factor), factor
                    )

            pseudo_inverse = (
                tl.reshape(weights, (-1, 1))
                * pseudo_inverse
                * tl.reshape(weights, (1, -1))
            )
            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)

            if mode in nn_modes:
                # Call the hals resolution with nnls, optimizing the current mode
                nn_factor = hals_nnls(
                    tl.transpose(mttkrp),
                    pseudo_inverse,
                    tl.transpose(factors[mode]),
                    n_iter_max=100,
                    sparsity_coefficient=sparsity_coefficients[mode],
                    exact=exact,
                )
                factors[mode] = tl.transpose(nn_factor)
            else:
                factor = tl.solve(tl.transpose(pseudo_inverse), tl.transpose(mttkrp))
                factors[mode] = tl.transpose(factor)
            if normalize_factors and mode != modes[-1]:
                weights, factors = cp_normalize((weights, factors))
        if tol:
            factors_norm = cp_norm((weights, factors))
            iprod = tl.sum(tl.sum(mttkrp * factors[-1], axis=0))
            rec_error = (
                tl.sqrt(tl.abs(norm_tensor**2 + factors_norm**2 - 2 * iprod))
                / norm_tensor
            )
            rec_errors.append(rec_error)
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print(
                        f"iteration {iteration}, reconstruction error: {rec_error}, decrease = {rec_error_decrease}"
                    )

                if cvg_criterion == "abs_rec_error":
                    stop_flag = tl.abs(rec_error_decrease) < tol
                elif cvg_criterion == "rec_error":
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print(f"PARAFAC converged after {iteration} iterations")
                    break
            else:
                if verbose:
                    print(f"reconstruction error={rec_errors[-1]}")
        if normalize_factors:
            weights, factors = cp_normalize((weights, factors))
    cp_tensor = CPTensor((weights, factors))
    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor


class CP_NN(DecompositionMixin):
    """
    Non-Negative Candecomp-Parafac decomposition via Alternating-Least Square

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [|weights; factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random'}, optional
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
        Stopping criterion for ALS, works if `tol` is not None.
        If 'rec_error',  ALS stops at current iteration if (previous rec_error - current rec_error) < tol.
        If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
    sparsity : float or int
        If `sparsity` is not None, we approximate tensor as a sum of low_rank_component and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). `sparsity` denotes desired fraction or number of non-zero elements in the sparse_component of the `tensor`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    svd_mask_repeats: int
        If using a tensor with masked values, this initializes using SVD multiple times to
        remove the effect of these missing values on the initialization.

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
            all ones if normalize_factors is False (default),
            weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape
            (tensor.shape[i], rank)
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
           SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values."
           Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.
    .. [3] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and
           Applications", PhD., University of Amsterdam, 1998
    """

    def __init__(
        self,
        rank,
        n_iter_max=100,
        init="svd",
        svd="truncated_svd",
        tol=10e-7,
        random_state=None,
        verbose=0,
        normalize_factors=False,
        mask=None,
        cvg_criterion="abs_rec_error",
        fixed_modes=None,
    ):
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.normalize_factors = normalize_factors
        self.mask = mask
        self.cvg_criterion = cvg_criterion
        self.fixed_modes = fixed_modes

    def fit_transform(self, tensor):
        """Decompose an input tensor

        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose

        Returns
        -------
        CPTensor
            decomposed tensor
        """

        cp_tensor, errors = non_negative_parafac(
            tensor,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            normalize_factors=self.normalize_factors,
            mask=self.mask,
            cvg_criterion=self.cvg_criterion,
            fixed_modes=self.fixed_modes,
            return_errors=True,
        )

        self.decomposition_ = cp_tensor
        self.errors_ = errors
        return self.decomposition_

    def __repr__(self):
        return f"Rank-{self.rank} Non-Negative CP decomposition."


class CP_NN_HALS(DecompositionMixin):
    """
    Non-Negative Candecomp-Parafac decomposition via Alternating-Least Square

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::

        ``tensor = [|weights; factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random'}, optional
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
        Stopping criterion for ALS, works if `tol` is not None.
        If 'rec_error',  ALS stops at current iteration if (previous rec_error - current rec_error) < tol.
        If 'abs_rec_error', ALS terminates when ``|previous rec_error - current rec_error| < tol``.
    sparsity : float or int
        If `sparsity` is not None, we approximate tensor as a sum of low_rank_component and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). `sparsity` denotes desired fraction or number of non-zero elements in the sparse_component of the `tensor`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    svd_mask_repeats: int
        If using a tensor with masked values, this initializes using SVD multiple times to
        remove the effect of these missing values on the initialization.

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
                all ones if normalize_factors is False (default),
                weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape
                (tensor.shape[i], rank)
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
           SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values."
           Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.
    .. [3] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and
           Applications", PhD., University of Amsterdam, 1998
    """

    def __init__(
        self,
        rank,
        n_iter_max=100,
        init="svd",
        svd="truncated_svd",
        tol=10e-8,
        sparsity_coefficients=None,
        fixed_modes=None,
        nn_modes="all",
        exact=False,
        verbose=False,
        normalize_factors=False,
        cvg_criterion="abs_rec_error",
        random_state=None,
    ):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.tol = tol
        self.sparsity_coefficients = sparsity_coefficients
        self.random_state = random_state
        self.fixed_modes = fixed_modes
        self.nn_modes = nn_modes
        self.exact = exact
        self.verbose = verbose
        self.normalize_factors = normalize_factors
        self.cvg_criterion = cvg_criterion
        self.random_state = random_state

    def fit_transform(self, tensor):
        """Decompose an input tensor

        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose

        Returns
        -------
        CPTensor
            decomposed tensor
        """

        cp_tensor, errors = non_negative_parafac_hals(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            tol=self.tol,
            random_state=self.random_state,
            sparsity_coefficients=self.sparsity_coefficients,
            fixed_modes=self.fixed_modes,
            nn_modes=self.nn_modes,
            exact=self.exact,
            verbose=self.verbose,
            normalize_factors=self.normalize_factors,
            return_errors=True,
            cvg_criterion=self.cvg_criterion,
        )

        self.decomposition_ = cp_tensor
        self.errors_ = errors
        return self.decomposition_

    def __repr__(self):
        return f"Rank-{self.rank} Non-Negative CP decomposition."
