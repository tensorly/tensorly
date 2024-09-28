from warnings import warn
from typing import Iterable, Optional, Sequence, Literal, Union

import tensorly as tl
from ._base_decomposition import DecompositionMixin
from tensorly.random import random_parafac2
from tensorly import backend as T
from . import parafac, non_negative_parafac_hals
from ..parafac2_tensor import (
    Parafac2Tensor,
    _validate_parafac2_tensor,
)
from ..cp_tensor import CPTensor, cp_normalize
from ..tenalg.svd import svd_interface, SVD_TYPES

# Authors: Marie Roald
#          Yngve Mardal Moe


def initialize_decomposition(
    tensor_slices, rank, init="random", svd="truncated_svd", random_state=None
):
    r"""Initiate a random PARAFAC2 decomposition given rank and tensor slices.

    The SVD-based initialization is based on concatenation of all the tensor slices.
    This concatenated matrix is used to derive the factor matrix corresponding to the
    :math:`k` mode for an :math:`X_{ijk}` tensor. However, concatenating these slices
    requires a new copy of the tensor. For tensors where the sum of the :math:`j` mode
    along each slice is on average larger than the :math:`k` mode, an alternative
    strategy is adding together the cross-product matrix of each slice:

    .. math::

        K = X_{1}^T X_{1} + X_{2}^T X_{2} + ...

    The eigenvectors of this symmetric matrix are then equal to the right eigenvectors
    of the concatenation matrix. This function automatically chooses between
    concatenating or forming the cross-product, depending on which resulting matrix
    is smaller.

    Parameters
    ----------
    tensor_slices : Iterable of ndarray
    rank : int
    init : {'random', 'svd', CPTensor, Parafac2Tensor}, optional
    random_state : `np.random.RandomState`

    Returns
    -------
    parafac2_tensor : Parafac2Tensor
        List of initialized factors of the CP decomposition where element `i`
        is of shape (tensor.shape[i], rank)
    """
    context = tl.context(tensor_slices[0])
    shapes = [m.shape for m in tensor_slices]
    concat_shape = sum(shape[0] for shape in shapes)

    if init == "random":
        return random_parafac2(
            shapes, rank, full=False, random_state=random_state, **context
        )
    elif init == "svd":
        if shapes[0][1] < rank:
            raise ValueError(
                f"Cannot perform SVD init if rank ({rank}) is greater than the number of columns in each tensor slice ({shapes[0][1]})"
            )

        A = tl.ones((len(tensor_slices), rank), **context)

        if concat_shape > shapes[0][1]:
            # If the concatenated matrix would be larger than the cross-product, use the latter
            unfolded_mode_2 = tl.transpose(tensor_slices[0]) @ tensor_slices[0]

            for slice in tensor_slices[1:]:
                unfolded_mode_2 += tl.matmul(tl.transpose(slice), slice)
        else:
            unfolded_mode_2 = tl.transpose(tl.concatenate(list(tensor_slices), axis=0))

        C = svd_interface(unfolded_mode_2, n_eigenvecs=rank, method=svd)[0]

        B = tl.eye(rank, **context)
        projections = _compute_projections(tensor_slices, (A, B, C), svd)
        return Parafac2Tensor((None, [A, B, C], projections))

    elif isinstance(init, (tuple, list, Parafac2Tensor, CPTensor)):
        try:
            decomposition = Parafac2Tensor.from_CPTensor(init, parafac2_tensor_ok=True)
        except ValueError:
            raise ValueError(
                "If initialization method is a mapping, then it must "
                "be possible to convert it to a Parafac2Tensor instance"
            )
        if decomposition.rank != rank:
            raise ValueError("Cannot init with a decomposition of different rank")
        return decomposition
    raise ValueError(f'Initialization method "{init}" not recognized')


def _compute_projections(tensor_slices, factors, svd):
    n_eig = factors[0].shape[1]
    out = []

    for A, tensor_slice in zip(factors[0], tensor_slices):
        lhs = T.dot(factors[1], T.transpose(A * factors[2]))
        rhs = T.transpose(tensor_slice)
        U, _, Vh = svd_interface(
            T.dot(lhs, rhs), n_eigenvecs=n_eig, method=svd, flip_sign=False
        )

        out.append(T.transpose(T.dot(U, Vh)))

    return out


def _project_tensor_slices(tensor_slices, projections):
    slices = []

    for t, p in zip(tensor_slices, projections):
        slices.append(T.dot(T.transpose(p), t))

    return tl.stack(slices)


class _BroThesisLineSearch:
    def __init__(
        self,
        norm_tensor,
        svd: str,
        verbose: bool = False,
        nn_modes=None,
        acc_pow: float = 2.0,
        max_fail: int = 4,
    ):
        """The line search strategy defined within Rasmus Bro's thesis [1, 2].

        Parameters
        ----------
        norm_tensor : int
            Sum of the matrix norms for each slice.
        svd : str
            The function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
        verbose : bool
            Optionally provide output about each step.
        nn_modes: None, 'all' or array of integers
            (Default: None) Used to specify which modes to impose non-negativity constraints on.
            We cannot impose non-negativity constraints on the the B-mode (mode 1) with the ALS
            algorithm, so if this mode is among the constrained modes, then a warning will be shown
            (see notes for more info).
        acc_pow : int
            Line search steps are defined as `iteration ** (1.0 / acc_pow)`.
        max_fail : int
            The number of line search failures before increasing `acc_pow`.

        References
        ----------
        .. [1] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and
            Applications", PhD., University of Amsterdam, 1998
        .. [2] H. Yu, D. Augustijn, R. Bro, "Accelerating PARAFAC2 algorithms for non-negative
            complex tensor decomposition." Chemometrics and Intelligent Laboratory Systems 214
            (2021): 104312.
        """
        self.norm_tensor = norm_tensor
        self.svd = svd
        self.verbose = verbose
        self.acc_pow = acc_pow  # Extrapolate to the iteration^(1/acc_pow) ahead
        self.max_fail = max_fail  # Increase acc_pow with one after max_fail failure
        self.acc_fail = 0  # How many times acceleration have failed
        self.nn_modes = nn_modes

    def line_step(
        self,
        iteration: int,
        tensor_slices: Iterable,
        factors_last: list,
        weights,
        factors: list,
        projections: list,
        rec_error,
    ):
        r"""Perform one line search step.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        tensor_slices : ndarray or list of ndarrays
            The data itself. Either a third order tensor or a list of second order tensors that
            may have different number of rows.
        factors_last : list of ndarrays
            The CP factors from the previous iteration.
        weights : ndarrays
            The normalization weights for the current factors.
        factors : list of ndarrays
            The CP factors from the current iteration.
        projections : list of ndarrays
            The projection matrices from the current iteration.
        rec_error : float
            The reconstruction error from the current iteration.

        Returns
        -------
        factors : list
            List of factors for the accepted step.
        projections : list
            List of projection matrices from the accepted step.
        rec_error : float
            Reconstruction error of the accepted step.
        """
        jump = iteration ** (1.0 / self.acc_pow)

        factors_ls = [
            factors_last[ii] + (factors[ii] - factors_last[ii]) * jump
            for ii, _ in enumerate(factors)
        ]

        # Clip if the mode should be non-negative
        if self.nn_modes:
            if 0 in self.nn_modes:
                factors_ls[0] = tl.clip(factors_ls[0], 0)
            if 2 in self.nn_modes:
                factors_ls[2] = tl.clip(factors_ls[2], 0)

        projections_ls = _compute_projections(tensor_slices, factors_ls, self.svd)

        ls_rec_error = _parafac2_reconstruction_error(
            tensor_slices, (weights, factors_ls, projections_ls), self.norm_tensor
        )
        ls_rec_error /= self.norm_tensor

        if ls_rec_error < rec_error:
            self.acc_fail = 0

            if self.verbose:
                print(f"Accepted line search jump of {jump}.")

            return factors_ls, projections_ls, ls_rec_error
        else:
            self.acc_fail += 1

            if self.verbose:
                print(f"Line search failed for jump of {jump}.")

            if self.acc_fail == self.max_fail:
                self.acc_pow += 1.0
                self.acc_fail = 0

                if self.verbose:
                    print("Reducing acceleration.")

            return factors, projections, rec_error


def _parafac2_reconstruction_error(
    tensor_slices, decomposition, norm_matrices=None, projected_tensor=None
):
    """Calculates the reconstruction error of the PARAFAC2 decomposition. This implementation
    uses the inner product with each matrix for efficiency, as this avoids needing to
    reconstruct the tensor. This is based on the property that:

    .. math::

        ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>

    Parameters
    ----------
    tensor_slices : ndarray or list of ndarrays
        The data itself. Either a third order tensor or a list of second order tensors that
        may have different number of rows.
    decomposition : (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            weights of the (normalized) factors
        * factors : List of factors of the CP decomposition element `i` is of shape
            (tensor.shape[i], rank)
        * projections : List of projection matrices used to create evolving factors.
    norm_matrices : float, optional
        The norm of the data. This can be optionally provided to avoid recalculating it.
    projected_tensor : ndarray, optional
        The projections of X into an aligned tensor for CP decomposition. This can be optionally
        provided to avoid recalculating it.

    Returns
    -------
    error : float
        The norm of the reconstruction error of the PARAFAC2 decomposition.
    """
    _validate_parafac2_tensor(decomposition)

    if norm_matrices is None:
        norm_X_sq = sum(tl.norm(t_slice, 2) ** 2 for t_slice in tensor_slices)
    else:
        norm_X_sq = norm_matrices**2

    weights, (A, B, C), projections = decomposition
    if weights is not None:
        A = A * weights

    norm_cmf_sq = 0
    inner_product = 0
    CtC = tl.dot(tl.transpose(C), C)

    for i, t_slice in enumerate(tensor_slices):
        B_i = (projections[i] @ B) * A[i]

        if projected_tensor is None:
            tmp = tl.dot(tl.transpose(B_i), t_slice)
        else:
            tmp = tl.reshape(A[i], (-1, 1)) * tl.transpose(B) @ projected_tensor[i]

        inner_product += tl.trace(tl.dot(tmp, C))

        norm_cmf_sq += tl.sum((tl.transpose(B_i) @ B_i) * CtC)

    return tl.sqrt(norm_X_sq - 2 * inner_product + norm_cmf_sq)


def parafac2(
    tensor_slices,
    rank: int,
    n_iter_max: int = 2000,
    init="random",
    svd: SVD_TYPES = "truncated_svd",
    normalize_factors: bool = False,
    tol: float = 1.0e-8,
    nn_modes: Optional[Union[Sequence[int], Literal["all"]]] = None,
    random_state=None,
    verbose: Union[bool, int] = False,
    return_errors: bool = False,
    n_iter_parafac: int = 5,
    linesearch: bool = True,
):
    r"""PARAFAC2 decomposition [1]_ of a third order tensor via alternating least squares (ALS)

    Computes a rank-`rank` PARAFAC2 decomposition of the third-order tensor defined by
    `tensor_slices`. The decomposition is on the form :math:`(A [B_i] C)` such that the
    i-th frontal slice, :math:`X_i`, of :math:`X` is given by

    .. math::

        X_i = B_i diag(a_i) C^T,

    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i`
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix.
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.


    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},

    with the same constraints hold for :math:`B_i` as above.


    Parameters
    ----------
    tensor_slices : ndarray or list of ndarrays
        Either a third order tensor or a list of second order tensors that may have different number of rows.
        Note that the second mode factor matrices are allowed to change over the first mode, not the
        third mode as some other implementations use (see note below).
    rank : int
        Number of components.
    n_iter_max : int, optional
        (Default: 2000) Maximum number of iteration

        .. versionchanged:: 0.6.1

            Previously, the default maximum number of iterations was 100.
    init : {'svd', 'random', CPTensor, Parafac2Tensor}
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : bool (optional)
        If True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors. Note that
        there may be some inaccuracies in the component weights.
    tol : float, optional
        (Default: 1e-8) Relative reconstruction error decrease tolerance. The
        algorithm is considered to have converged when
        :math:`\left|\| X - \hat{X}_{n-1} \|^2 - \| X - \hat{X}_{n} \|^2\right| < \epsilon \| X - \hat{X}_{n-1} \|^2`.
        That is, when the relative change in sum of squared error is less
        than the tolerance.

        .. versionchanged:: 0.6.1

            Previously, the stopping condition was
            :math:`\left|\| X - \hat{X}_{n-1} \| - \| X - \hat{X}_{n} \|\right| < \epsilon`.
    nn_modes: None, 'all' or array of integers
        (Default: None) Used to specify which modes to impose non-negativity constraints on.
        We cannot impose non-negativity constraints on the the B-mode (mode 1) with the ALS
        algorithm, so if this mode is among the constrained modes, then a warning will be shown
        (see notes for more info).
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    n_iter_parafac : int, optional
        Number of PARAFAC iterations to perform for each PARAFAC2 iteration
    linesearch : bool, default is False
        Whether to perform line search as proposed by Bro in his PhD dissertation [2]_
        (similar to the PLSToolbox line search described in [3]_).

    Returns
    -------
    Parafac2Tensor : (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            all ones if normalize_factors is False (default),
            weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape
            (tensor.shape[i], rank)
        * projection_matrices : List of projection matrices used to create evolving
            factors.

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] Kiers, H.A.L., ten Berge, J.M.F. and Bro, R. (1999),
            PARAFAC2—Part I. A direct fitting algorithm for the PARAFAC2 model.
            J. Chemometrics, 13: 275-294.
    .. [2] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and
            Applications", PhD., University of Amsterdam, 1998
    .. [3] H. Yu, D. Augustijn, R. Bro, "Accelerating PARAFAC2 algorithms for non-negative
            complex tensor decomposition." Chemometrics and Intelligent Laboratory Systems 214
            (2021): 104312.

    Notes
    -----
    This formulation of the PARAFAC2 decomposition is slightly different from the one in [1]_.
    The difference lies in that, here, the second mode changes over the first mode, whereas in
    [1]_, the second mode changes over the third mode. This change allows the function to accept
    both lists of matrices and a single nd-array as input without any mode reordering.

    Because of the reformulation above, :math:`B_i = P_i B`, the :math:`B_i` matrices
    cannot be constrained to be non-negative with ALS. If this mode is constrained to be
    non-negative, then :math:`B` will be non-negative, but not the orthogonal `P_i` matrices.
    Consequently, the `B_i` matrices are unlikely to be non-negative.
    """
    assert (
        rank <= tensor_slices[0].shape[1]
    ), f"PARAFAC2 rank ({rank}) cannot be greater than the number of columns in each tensor slice ({tensor_slices[0].shape[1]})."

    for ii in range(1, len(tensor_slices)):
        assert (
            tensor_slices[0].shape[1] == tensor_slices[ii].shape[1]
        ), "All tensor slices must have the same number of columns."

    weights, factors, projections = initialize_decomposition(
        tensor_slices, rank, init=init, svd=svd, random_state=random_state
    )
    factors = list(factors)

    rec_errors = []
    norm_tensor = tl.sqrt(
        sum(tl.norm(tensor_slice, 2) ** 2 for tensor_slice in tensor_slices)
    )

    if linesearch and not isinstance(linesearch, _BroThesisLineSearch):
        linesearch = _BroThesisLineSearch(
            norm_tensor, svd, verbose=verbose, nn_modes=nn_modes
        )

    # If nn_modes is set, we use HALS, otherwise, we use the standard parafac implementation.
    if nn_modes is None:

        def parafac_updates(X, w, f):
            return parafac(
                X,
                rank,
                n_iter_max=n_iter_parafac,
                init=(w, f),
                svd=svd,
                orthogonalise=False,
                verbose=verbose,
                return_errors=False,
                normalize_factors=False,
                mask=None,
                random_state=random_state,
                tol=1e-100,
            )[1]

    else:
        if nn_modes == "all" or 1 in nn_modes:
            warn(
                "Mode `1` of PARAFAC2 fitted with ALS cannot be constrained to be truly non-negative. See the documentation for more info."
            )

        def parafac_updates(X, w, f):
            return non_negative_parafac_hals(
                X,
                rank,
                n_iter_max=n_iter_parafac,
                init=(w, f),
                svd=svd,
                nn_modes=nn_modes,
                verbose=verbose,
                return_errors=False,
                tol=1e-100,
            )[1]

    for iteration in range(n_iter_max):
        if verbose:
            print("Starting iteration", iteration)

        factors[1] = factors[1] * T.reshape(weights, (1, -1))
        weights = T.ones(weights.shape, **tl.context(tensor_slices[0]))

        # Will we be performing a line search iteration?
        if linesearch and iteration % 2 == 0 and iteration > 5:
            line_iter = True
            factors_last = [tl.copy(f) for f in factors]
        else:
            line_iter = False

        projections = _compute_projections(tensor_slices, factors, svd)
        projected_tensor = _project_tensor_slices(tensor_slices, projections)
        factors = parafac_updates(projected_tensor, weights, factors)

        # Start line search if requested.
        if line_iter:
            factors, projections, rec_errors[-1] = linesearch.line_step(
                iteration,
                tensor_slices,
                factors_last,
                weights,
                factors,
                projections,
                rec_errors[-1],
            )

        if normalize_factors:
            weights, factors = cp_normalize((weights, factors))

        if tol and not line_iter:
            rec_error = _parafac2_reconstruction_error(
                tensor_slices,
                (weights, factors, projections),
                norm_tensor,
                projected_tensor,
            )
            rec_error /= norm_tensor
            rec_errors.append(rec_error)

        if tol:
            if iteration >= 1:
                if verbose:
                    print(
                        f"PARAFAC2 reconstruction error={rec_errors[-1]}, variation={rec_errors[-2] - rec_errors[-1]}."
                    )

                if tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    if verbose:
                        print(f"converged in {iteration} iterations.")
                    break
            else:
                if verbose:
                    print(f"PARAFAC2 reconstruction error={rec_errors[-1]}")

    parafac2_tensor = Parafac2Tensor((weights, factors, projections))

    if return_errors:
        return parafac2_tensor, rec_errors
    else:
        return parafac2_tensor


class Parafac2(DecompositionMixin):
    r"""PARAFAC2 decomposition [1]_ of a third order tensor via alternating least squares (ALS)

    Computes a rank-`rank` PARAFAC2 decomposition of the third-order tensor defined by
    `tensor_slices`. The decomposition is on the form :math:`(A [B_i] C)` such that the
    i-th frontal slice, :math:`X_i`, of :math:`X` is given by

    .. math::

        X_i = B_i diag(a_i) C^T,

    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i`
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix.
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.

    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},

    with the same constraints hold for :math:`B_i` as above.

    Parameters
    ----------
    rank : int
        Number of components.
    n_iter_max : int, optional
        (Default: 2000) Maximum number of iteration

        .. versionchanged:: 0.6.1

            Previously, the default maximum number of iterations was 100.

    init : {'svd', 'random', CPTensor, Parafac2Tensor}
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : bool (optional)
        If True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors. Note that
        there may be some inaccuracies in the component weights.
    tol : float, optional
        (Default: 1e-8) Relative reconstruction error decrease tolerance. The
        algorithm is considered to have converged when
        :math:`\left|\| X - \hat{X}_{n-1} \|^2 - \| X - \hat{X}_{n} \|^2\right| < \epsilon \| X - \hat{X}_{n-1} \|^2`.
        That is, when the relative change in sum of squared error is less
        than the tolerance.

        .. versionchanged:: 0.6.1

            Previously, the stopping condition was
            :math:`\left|\| X - \hat{X}_{n-1} \| - \| X - \hat{X}_{n} \|\right| < \epsilon`.
    nn_modes: None, 'all' or array of integers
        (Default: None) Used to specify which modes to impose non-negativity constraints on.
        We cannot impose non-negativity constraints on the the B-mode (mode 1) with the ALS
        algorithm, so if this mode is among the constrained modes, then a warning will be shown
        (see notes for more info).
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    n_iter_parafac : int, optional
        Number of PARAFAC iterations to perform for each PARAFAC2 iteration

    Returns
    -------
    Parafac2Tensor : (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            all ones if normalize_factors is False (default),
            weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape
            (tensor.shape[i], rank)
        * projection_matrices : List of projection matrices used to create evolving
            factors.

    References
    ----------
    .. [1] Kiers, H.A.L., ten Berge, J.M.F. and Bro, R. (1999),
           PARAFAC2—Part I. A direct fitting algorithm for the PARAFAC2 model.
           J. Chemometrics, 13: 275-294.

    Notes
    -----
    This formulation of the PARAFAC2 decomposition is slightly different from the one in [1]_.
    The difference lies in that, here, the second mode changes over the first mode, whereas in
    [1]_, the second mode changes over the third mode. This change allows the function to accept
    both lists of matrices and a single nd-array as input without any mode reordering.
    """

    def __init__(
        self,
        rank,
        n_iter_max=2000,
        init="random",
        svd="truncated_svd",
        normalize_factors=False,
        tol=1e-8,
        nn_modes=None,
        random_state=None,
        verbose=False,
        return_errors=False,
        n_iter_parafac=5,
        linesearch=False,
    ):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.normalize_factors = normalize_factors
        self.tol = tol
        self.nn_modes = nn_modes
        self.random_state = random_state
        self.verbose = verbose
        self.return_errors = return_errors
        self.n_iter_parafac = n_iter_parafac
        self.linesearch = linesearch

    def fit_transform(self, tensor):
        """Decompose an input tensor

        Parameters
        ----------
        tensor : tensorly.tensor

        Returns
        -------
        self
        """
        self.decomposition_, self.errors_ = parafac2(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            normalize_factors=self.normalize_factors,
            tol=self.tol,
            nn_modes=self.nn_modes,
            random_state=self.random_state,
            verbose=self.verbose,
            return_errors=self.return_errors,
            n_iter_parafac=self.n_iter_parafac,
            linesearch=self.linesearch,
        )
        return self.decomposition_
