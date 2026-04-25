import warnings
import numpy as np

import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..base import unfold, fold
from ..tenalg.svd import svd_interface


# Authors: TensorLy Contributors
# License: BSD 3 clause


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ll1_to_tensor(matrices, vectors):
    """Reconstruct a tensor from an LL1 decomposition.

    Computes ``T[i, j, k] = sum_r matrices[r][i, j] * vectors[k, r]``.

    Parameters
    ----------
    matrices : list of ndarray
        R activation matrices, each of shape ``(I, J)``.
    vectors : ndarray of shape ``(K, R)``
        Mode-K factor matrix whose columns are the Stokes/mode-K vectors.

    Returns
    -------
    tensor : ndarray of shape ``(I, J, K)``
    """
    I, J = tl.shape(matrices[0])
    K, R = tl.shape(vectors)
    ctx = tl.context(vectors)

    result = tl.zeros((I, J, K), **ctx)
    for r in range(R):
        term = tl.reshape(matrices[r], (I, J, 1)) * tl.reshape(
            vectors[:, r], (1, 1, K)
        )
        result = result + term
    return result


def _proj_stokes_matrix(C):
    r"""Project each column of *C* onto the Stokes (Lorentz) cone.

    A valid Stokes vector ``[S0, S1, S2, S3]`` satisfies the second-order
    (Lorentz / ice-cream) cone constraint::

        S0 >= ||(S1, S2, S3)||_2  and  S0 >= 0.

    The projection of an arbitrary point ``(t, x)`` onto the cone is:

    * If ``t >= ||x||``: already feasible – return the point unchanged.
    * If ``t <= -||x||``: point is in the *opposite* cone – return **0**.
    * Otherwise: project to the boundary:
      ``t_new = (t + ||x||) / 2``,
      ``x_new = (t_new / ||x||) * x``.

    Parameters
    ----------
    C : ndarray of shape ``(K, R)``
        Matrix whose columns are to be projected.  Must have ``K >= 2``
        (the first row is treated as *t* and the remaining rows as *x*).

    Returns
    -------
    C_projected : ndarray of shape ``(K, R)``
    """
    ctx = tl.context(C)
    C_np = np.array(tl.to_numpy(C), dtype=np.float64)
    K, R = C_np.shape

    for r in range(R):
        col = C_np[:, r]
        t = col[0]
        x = col[1:]
        x_norm = np.linalg.norm(x)

        if t >= x_norm:
            pass  # already in cone
        elif t <= -x_norm:
            C_np[:, r] = 0.0  # anti-cone → project to zero
        else:
            t_new = (x_norm + t) / 2.0
            if x_norm > 1e-14:
                C_np[0, r] = t_new
                C_np[1:, r] = (t_new / x_norm) * x
            else:
                C_np[0, r] = t_new
                C_np[1:, r] = 0.0

    return tl.tensor(C_np, **ctx)


# ---------------------------------------------------------------------------
# Public API – uniqueness check
# ---------------------------------------------------------------------------

def check_ll1_uniqueness(shape, rank):
    """Check necessary conditions for generic uniqueness of the LL1 decomposition.

    The LL1 model represents a 3rd-order tensor ``T`` of shape ``(I, J, K)``
    as a sum of *R* terms

    .. math::

        \\mathcal{T} = \\sum_{r=1}^{R} \\mathbf{A}_r \\otimes \\mathbf{c}_r,

    where :math:`\\mathbf{A}_r` is an ``(I, J)`` activation matrix and
    :math:`\\mathbf{c}_r` is a *K*-vector.

    The decomposition can be generically identifiable only if

    * ``K >= R`` – the mode-*K* dimension is large enough to accommodate *R*
      linearly independent vectors.
    * ``I * J >= R`` – there are enough degrees of freedom for *R* independent
      activation matrices.

    Parameters
    ----------
    shape : tuple of int
        Shape ``(I, J, K)`` of the tensor.
    rank : int
        Number of terms *R* in the LL1 decomposition.

    Returns
    -------
    bool
        ``True`` if the necessary dimensional conditions are satisfied,
        ``False`` otherwise.

    Notes
    -----
    These are *necessary* but not *sufficient* conditions.  Generic uniqueness
    additionally requires algebraic conditions on the factor matrices.

    References
    ----------
    .. [1] L. De Lathauwer, "Decompositions of a Higher-Order Tensor in Block
           Terms – Part II: Definitions and Uniqueness," *SIAM J. Matrix Anal.
           Appl.*, 30(3):1033–1066, 2008.
    .. [2] L. Sorber, M. Van Barel, L. De Lathauwer, "Structured Data
           Fusion," *IEEE J. Sel. Topics Signal Process.*, 9(4):586–600, 2015.
    """
    if len(shape) != 3:
        raise ValueError(
            f"LL1 uniqueness check requires a 3rd-order tensor shape (I, J, K), "
            f"got {shape}."
        )
    I, J, K = shape
    if K < rank:
        warnings.warn(
            f"Uniqueness condition K >= R is violated: K={K} < R={rank}. "
            "The LL1 decomposition is not identifiable.",
            stacklevel=2,
        )
        return False
    if I * J < rank:
        warnings.warn(
            f"Uniqueness condition I*J >= R is violated: I*J={I * J} < R={rank}. "
            "The LL1 decomposition is not identifiable.",
            stacklevel=2,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_ll1(tensor, rank, init="svd", svd="truncated_svd", random_state=None):
    """Initialize the LL1 decomposition factors.

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, K)``
        Input tensor.
    rank : int
        Number of LL1 terms *R*.
    init : ``{'svd', 'random'}`` or 2-tuple ``(matrices, vectors)``, optional
        Initialization method.  Default is ``'svd'``.
    svd : str, optional
        SVD algorithm to use.  Default is ``'truncated_svd'``.
    random_state : None, int, or ``RandomState``, optional

    Returns
    -------
    matrices : list of ndarray
        *R* activation matrices, each of shape ``(I, J)``.
    vectors : ndarray of shape ``(K, R)``
        Initial mode-*K* factor matrix.
    """
    rng = tl.check_random_state(random_state)
    I, J, K = tl.shape(tensor)
    ctx = tl.context(tensor)

    if isinstance(init, (tuple, list)) and len(init) == 2:
        matrices_init, vectors_init = init
        if len(matrices_init) != rank:
            raise ValueError(
                f"Provided initialization contains {len(matrices_init)} "
                f"matrices but rank={rank}."
            )
        return list(matrices_init), vectors_init

    if init == "random":
        matrices = [
            tl.tensor(rng.random_sample((I, J)), **ctx) for _ in range(rank)
        ]
        vectors = tl.tensor(rng.random_sample((K, rank)), **ctx)

    elif init == "svd":
        T_unf = unfold(tensor, 2)  # K × IJ
        n_sv = min(rank, min(K, I * J))
        U, S, Vt = svd_interface(T_unf, n_eigenvecs=n_sv, method=svd)
        # U: K×n_sv,  S: n_sv,  Vt: n_sv×IJ
        sqrt_s = tl.sqrt(tl.abs(S[:n_sv]))

        # vectors = U * sqrt(S) →  K×n_sv
        vectors = U[:, :n_sv] * sqrt_s[None, :]

        # A_flat_row = Vt * sqrt(S)  →  n_sv×IJ  → transpose: IJ×n_sv
        A_flat = tl.transpose(Vt[:n_sv, :]) * sqrt_s[None, :]  # IJ×n_sv

        if n_sv < rank:
            # Pad with small random values when rank > min_dim
            extra_K = tl.tensor(
                rng.random_sample((K, rank - n_sv)) * 1e-3, **ctx
            )
            vectors = tl.concatenate([vectors, extra_K], axis=1)
            extra_A = tl.tensor(
                rng.random_sample((I * J, rank - n_sv)) * 1e-3, **ctx
            )
            A_flat = tl.concatenate([A_flat, extra_A], axis=1)

        matrices = [tl.reshape(A_flat[:, r], (I, J)) for r in range(rank)]

    else:
        raise ValueError(f'Initialization method "{init}" not recognized.')

    return matrices, vectors


# ---------------------------------------------------------------------------
# Algorithm 1 – Alternating Least Squares (unconstrained)
# ---------------------------------------------------------------------------

def ll1_als(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
    tol=1e-8,
    random_state=None,
    verbose=False,
    return_errors=False,
):
    r"""LL1 decomposition via Alternating Least Squares (ALS).

    Fits the model

    .. math::

        \\mathcal{T} \\approx \\sum_{r=1}^{R} \\mathbf{A}_r \\otimes \\mathbf{c}_r

    by alternating between closed-form least-squares updates of the
    activation matrices :math:`\\{\\mathbf{A}_r\\}` and the mode-*K* vectors
    :math:`\\{\\mathbf{c}_r\\}`, operating on the mode-*K* unfolding.

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, K)``
    rank : int
        Number of LL1 terms *R*.
    n_iter_max : int, optional
        Maximum number of ALS iterations.  Default ``100``.
    init : ``{'svd', 'random'}`` or 2-tuple, optional
        Factor initialization.  Default ``'svd'``.
    svd : str, optional
        SVD method for ``'svd'`` initialization.  Default ``'truncated_svd'``.
    tol : float, optional
        Stop when ``|rec_error[t-1] - rec_error[t]| < tol``.
        Default ``1e-8``.
    random_state : None, int, or ``RandomState``, optional
    verbose : bool, optional
        Print reconstruction error at each iteration.  Default ``False``.
    return_errors : bool, optional
        Also return the per-iteration reconstruction error list.
        Default ``False``.

    Returns
    -------
    (matrices, vectors) : tuple
        *matrices* is a list of *R* arrays of shape ``(I, J)``.
        *vectors* is an array of shape ``(K, R)``.
    errors : list of float
        Only returned when ``return_errors=True``.

    References
    ----------
    .. [1] L. De Lathauwer, "Decompositions of a Higher-Order Tensor in Block
           Terms – Part II," *SIAM J. Matrix Anal. Appl.*, 30(3), 2008.
    """
    if tl.ndim(tensor) != 3:
        raise ValueError("ll1_als requires a 3rd-order (3-D) tensor.")

    I, J, K = tl.shape(tensor)
    ctx = tl.context(tensor)

    matrices, vectors = initialize_ll1(
        tensor, rank, init=init, svd=svd, random_state=random_state
    )

    norm_tensor = tl.norm(tensor)
    T_unf = unfold(tensor, 2)  # K × IJ

    rec_errors = []

    for iteration in range(n_iter_max):
        # Ã  matrix (IJ × R): column r = vec(A_r)
        A_flat = tl.stack(
            [tl.reshape(matrices[r], (I * J,)) for r in range(rank)], axis=1
        )  # IJ × R

        # ---- Update C (vectors) given Ã ----
        # C = T_unf @ Ã @ inv(Ã.T @ Ã)
        UtU_C = tl.dot(tl.transpose(A_flat), A_flat)   # R × R
        UtM_C = tl.dot(T_unf, A_flat)                  # K × R
        # Solve: UtU_C.T @ vectors.T = UtM_C.T
        vectors = tl.transpose(
            tl.solve(tl.transpose(UtU_C), tl.transpose(UtM_C))
        )  # K × R

        # ---- Update Ã (matrices) given C ----
        # Ã = T_unf.T @ C @ inv(C.T @ C)
        UtU_A = tl.dot(tl.transpose(vectors), vectors)    # R × R
        UtM_A = tl.dot(tl.transpose(T_unf), vectors)      # IJ × R
        A_flat = tl.transpose(
            tl.solve(tl.transpose(UtU_A), tl.transpose(UtM_A))
        )  # IJ × R

        matrices = [tl.reshape(A_flat[:, r], (I, J)) for r in range(rank)]

        # Reconstruction error
        T_rec = fold(tl.dot(vectors, tl.transpose(A_flat)), mode=2, shape=(I, J, K))
        rec_error = tl.norm(tensor - T_rec) / norm_tensor
        rec_errors.append(float(tl.to_numpy(rec_error)))

        if verbose:
            print(
                f"Iteration {iteration + 1:4d} | rec. error = {rec_errors[-1]:.6e}"
            )

        if iteration > 0 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print(f"Converged in {iteration + 1} iterations.")
            break

    result = (matrices, vectors)
    if return_errors:
        return result, rec_errors
    return result


# ---------------------------------------------------------------------------
# Algorithm 2 – Block-Proximal Gradient (constrained)
# ---------------------------------------------------------------------------

def ll1_bpg(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
    tol=1e-8,
    non_negative_matrices=True,
    stokes_vectors=True,
    random_state=None,
    verbose=False,
    return_errors=False,
):
    r"""LL1 decomposition via Block-Proximal Gradient (BPG).

    Fits the constrained model

    .. math::

        \\mathcal{T} \\approx \\sum_{r=1}^{R} \\mathbf{A}_r \\otimes \\mathbf{c}_r

    using projected alternating least squares (a.k.a. BPG with Lipschitz
    step-size).  Each block update consists of a gradient step with the
    exact Lipschitz constant followed by the corresponding projection:

    * **Activation matrices** :math:`\\{\\mathbf{A}_r\\}`: non-negativity
      projection ``max(0, ·)`` (when ``non_negative_matrices=True``).
    * **Stokes vectors** :math:`\\{\\mathbf{c}_r\\}`: projection onto the
      Lorentz / ice-cream cone (when ``stokes_vectors=True``).

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, K)``
    rank : int
    n_iter_max : int, optional
        Default ``100``.
    init : ``{'svd', 'random'}`` or 2-tuple, optional
    svd : str, optional
    tol : float, optional
    non_negative_matrices : bool, optional
        Enforce ``A_r >= 0``.  Default ``True``.
    stokes_vectors : bool, optional
        Enforce the Stokes cone constraint on each ``c_r``.  Requires
        ``K >= 2`` (typically ``K = 4``).  Default ``True``.
    random_state : None, int, or ``RandomState``, optional
    verbose : bool, optional
    return_errors : bool, optional

    Returns
    -------
    (matrices, vectors) : tuple
    errors : list (only when ``return_errors=True``)

    References
    ----------
    .. [1] L. De Lathauwer, "Decompositions of a Higher-Order Tensor in Block
           Terms – Part II," *SIAM J. Matrix Anal. Appl.*, 30(3), 2008.
    .. [2] J. Xu, "Alternating Proximal Gradient Method for Sparse Nonneg.
           Matrix Factorization," arXiv:1209.3916, 2012.
    """
    if tl.ndim(tensor) != 3:
        raise ValueError("ll1_bpg requires a 3rd-order (3-D) tensor.")

    I, J, K = tl.shape(tensor)
    ctx = tl.context(tensor)

    if stokes_vectors and K < 2:
        raise ValueError(
            f"Stokes vectors require K >= 2, but the tensor has K={K}."
        )

    matrices, vectors = initialize_ll1(
        tensor, rank, init=init, svd=svd, random_state=random_state
    )

    # Project initial factors onto constraints
    if non_negative_matrices:
        matrices = [tl.clip(m, a_min=0, a_max=None) for m in matrices]
    if stokes_vectors:
        vectors = _proj_stokes_matrix(vectors)

    norm_tensor = tl.norm(tensor)
    T_unf = unfold(tensor, 2)  # K × IJ

    rec_errors = []

    for iteration in range(n_iter_max):
        A_flat = tl.stack(
            [tl.reshape(matrices[r], (I * J,)) for r in range(rank)], axis=1
        )  # IJ × R

        # ---- Update C given Ã (with Stokes projection) ----
        UtU_C = tl.dot(tl.transpose(A_flat), A_flat)   # R × R
        UtM_C = tl.dot(T_unf, A_flat)                  # K × R
        vectors = tl.transpose(
            tl.solve(tl.transpose(UtU_C), tl.transpose(UtM_C))
        )  # K × R
        if stokes_vectors:
            vectors = _proj_stokes_matrix(vectors)

        # ---- Update Ã given C (with non-negativity projection) ----
        UtU_A = tl.dot(tl.transpose(vectors), vectors)    # R × R
        UtM_A = tl.dot(tl.transpose(T_unf), vectors)      # IJ × R
        A_flat = tl.transpose(
            tl.solve(tl.transpose(UtU_A), tl.transpose(UtM_A))
        )  # IJ × R
        if non_negative_matrices:
            A_flat = tl.clip(A_flat, a_min=0, a_max=None)

        matrices = [tl.reshape(A_flat[:, r], (I, J)) for r in range(rank)]

        # Reconstruction error
        T_rec = fold(tl.dot(vectors, tl.transpose(A_flat)), mode=2, shape=(I, J, K))
        rec_error = tl.norm(tensor - T_rec) / norm_tensor
        rec_errors.append(float(tl.to_numpy(rec_error)))

        if verbose:
            print(
                f"Iteration {iteration + 1:4d} | rec. error = {rec_errors[-1]:.6e}"
            )

        if iteration > 0 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print(f"Converged in {iteration + 1} iterations.")
            break

    result = (matrices, vectors)
    if return_errors:
        return result, rec_errors
    return result


# ---------------------------------------------------------------------------
# Algorithm 3 – AO-ADMM (constrained)
# ---------------------------------------------------------------------------

def ll1_ao_admm(
    tensor,
    rank,
    n_iter_max=100,
    n_iter_max_inner=10,
    init="svd",
    svd="truncated_svd",
    tol_outer=1e-8,
    tol_inner=1e-6,
    rho=None,
    non_negative_matrices=True,
    stokes_vectors=True,
    random_state=None,
    verbose=False,
    return_errors=False,
):
    r"""LL1 decomposition via Alternating Optimization ADMM (AO-ADMM).

    Fits the constrained model

    .. math::

        \\mathcal{T} \\approx \\sum_{r=1}^{R} \\mathbf{A}_r \\otimes \\mathbf{c}_r

    by alternating between ADMM sub-problems – one for the activation
    matrices and one for the Stokes vectors – each exploiting the mode-*K*
    unfolding.

    The ADMM sub-problem for a factor matrix **F** with constraint
    :math:`\\mathbf{F} \\in \\mathcal{C}` reads

    .. math::

        \\min_{\\mathbf{F}} \\|\\mathcal{T}_{(K)} - \\mathbf{U} \\mathbf{F}^T\\|_F^2
        \\quad \\text{s.t.} \\quad \\mathbf{F} \\in \\mathcal{C},

    which is solved via the scaled-form ADMM of [1]_.

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, K)``
    rank : int
    n_iter_max : int, optional
        Outer AO iterations.  Default ``100``.
    n_iter_max_inner : int, optional
        Inner ADMM iterations per factor update.  Default ``10``.
    init : ``{'svd', 'random'}`` or 2-tuple, optional
    svd : str, optional
    tol_outer : float, optional
        Outer convergence tolerance.  Default ``1e-8``.
    tol_inner : float, optional
        Inner ADMM convergence tolerance.  Default ``1e-6``.
    rho : float or None, optional
        ADMM penalty parameter.  If ``None``, set automatically as
        ``trace(UtU) / R``.
    non_negative_matrices : bool, optional
        Enforce ``A_r >= 0``.  Default ``True``.
    stokes_vectors : bool, optional
        Enforce the Stokes cone constraint on each ``c_r``.  Default ``True``.
    random_state : None, int, or ``RandomState``, optional
    verbose : bool, optional
    return_errors : bool, optional

    Returns
    -------
    (matrices, vectors) : tuple
    errors : list (only when ``return_errors=True``)

    References
    ----------
    .. [1] K. Huang, N. D. Sidiropoulos, A. P. Liavas, "A Flexible and
           Efficient Algorithmic Framework for Constrained Matrix and Tensor
           Factorization," *IEEE Trans. Signal Process.*, 64(19), 2016.
    .. [2] L. De Lathauwer, "Decompositions of a Higher-Order Tensor in Block
           Terms – Part II," *SIAM J. Matrix Anal. Appl.*, 30(3), 2008.
    """
    if tl.ndim(tensor) != 3:
        raise ValueError("ll1_ao_admm requires a 3rd-order (3-D) tensor.")

    I, J, K = tl.shape(tensor)
    ctx = tl.context(tensor)

    if stokes_vectors and K < 2:
        raise ValueError(
            f"Stokes vectors require K >= 2, but the tensor has K={K}."
        )

    matrices, vectors = initialize_ll1(
        tensor, rank, init=init, svd=svd, random_state=random_state
    )

    # Project initial factors
    if non_negative_matrices:
        matrices = [tl.clip(m, a_min=0, a_max=None) for m in matrices]
    if stokes_vectors:
        vectors = _proj_stokes_matrix(vectors)

    norm_tensor = tl.norm(tensor)
    T_unf = unfold(tensor, 2)  # K × IJ

    # Build initial A_flat
    A_flat = tl.stack(
        [tl.reshape(matrices[r], (I * J,)) for r in range(rank)], axis=1
    )  # IJ × R

    # ADMM dual variables (scaled form) – initialised to zero
    A_dual = tl.zeros((I * J, rank), **ctx)  # IJ × R
    C_dual = tl.zeros((K, rank), **ctx)      # K × R

    rec_errors = []

    for iteration in range(n_iter_max):

        # ================================================================
        # Update Ã (activation matrices) via ADMM
        # Problem: min_{Ã} ||T_unf - vectors @ Ã.T||_F^2   s.t. Ã ≥ 0
        # ================================================================
        UtU_A = tl.dot(tl.transpose(vectors), vectors)   # R × R
        UtM_A = tl.dot(tl.transpose(T_unf), vectors)     # IJ × R

        rho_A = float(tl.to_numpy(tl.trace(UtU_A))) / rank if rho is None else rho
        rho_A_t = tl.tensor(rho_A, **ctx)

        I_R_A = tl.eye(rank, **ctx)
        A_split = tl.zeros((rank, I * J), **ctx)   # R × IJ  (x_split, transposed)

        for _ in range(n_iter_max_inner):
            A_old = tl.copy(A_flat)

            # x_split = (UtU + rho*I)^{-1} (UtM + rho*(A_flat + A_dual)).T
            rhs_A = tl.transpose(UtM_A + rho_A_t * (A_flat + A_dual))  # R × IJ
            A_split = tl.solve(
                tl.transpose(UtU_A + rho_A_t * I_R_A), rhs_A
            )  # R × IJ

            # x update: project x_split.T - dual onto constraint
            A_unconstrained = tl.transpose(A_split) - A_dual  # IJ × R
            if non_negative_matrices:
                A_flat = tl.clip(A_unconstrained, a_min=0, a_max=None)
            else:
                A_flat = A_unconstrained

            # dual update
            A_dual = A_dual + A_flat - tl.transpose(A_split)

            # inner convergence check
            dual_res_A = tl.norm(A_flat - tl.transpose(A_split))
            primal_res_A = tl.norm(A_flat - A_old)
            norm_A = tl.norm(A_flat)
            norm_dual_A = tl.norm(A_dual)
            if (
                float(tl.to_numpy(dual_res_A))
                < tol_inner * float(tl.to_numpy(norm_A)) + 1e-14
                and float(tl.to_numpy(primal_res_A))
                < tol_inner * float(tl.to_numpy(norm_dual_A)) + 1e-14
            ):
                break

        matrices = [tl.reshape(A_flat[:, r], (I, J)) for r in range(rank)]

        # ================================================================
        # Update C (Stokes vectors) via ADMM
        # Problem: min_C ||T_unf - C @ Ã.T||_F^2   s.t. cols of C in Stokes cone
        # ================================================================
        UtU_C = tl.dot(tl.transpose(A_flat), A_flat)   # R × R
        UtM_C = tl.dot(T_unf, A_flat)                  # K × R

        rho_C = float(tl.to_numpy(tl.trace(UtU_C))) / rank if rho is None else rho
        rho_C_t = tl.tensor(rho_C, **ctx)

        I_R_C = tl.eye(rank, **ctx)
        C_split = tl.zeros((rank, K), **ctx)   # R × K

        for _ in range(n_iter_max_inner):
            C_old = tl.copy(vectors)

            # x_split update
            rhs_C = tl.transpose(UtM_C + rho_C_t * (vectors + C_dual))  # R × K
            C_split = tl.solve(
                tl.transpose(UtU_C + rho_C_t * I_R_C), rhs_C
            )  # R × K

            # x update: project x_split.T - dual onto Stokes cone
            C_unconstrained = tl.transpose(C_split) - C_dual  # K × R
            if stokes_vectors:
                vectors = _proj_stokes_matrix(C_unconstrained)
            else:
                vectors = C_unconstrained

            # dual update
            C_dual = C_dual + vectors - tl.transpose(C_split)

            # inner convergence check
            dual_res_C = tl.norm(vectors - tl.transpose(C_split))
            primal_res_C = tl.norm(vectors - C_old)
            norm_C = tl.norm(vectors)
            norm_dual_C = tl.norm(C_dual)
            if (
                float(tl.to_numpy(dual_res_C))
                < tol_inner * float(tl.to_numpy(norm_C)) + 1e-14
                and float(tl.to_numpy(primal_res_C))
                < tol_inner * float(tl.to_numpy(norm_dual_C)) + 1e-14
            ):
                break

        # Reconstruction error
        T_rec = fold(tl.dot(vectors, tl.transpose(A_flat)), mode=2, shape=(I, J, K))
        rec_error = tl.norm(tensor - T_rec) / norm_tensor
        rec_errors.append(float(tl.to_numpy(rec_error)))

        if verbose:
            print(
                f"Iteration {iteration + 1:4d} | rec. error = {rec_errors[-1]:.6e}"
            )

        if iteration > 0 and abs(rec_errors[-2] - rec_errors[-1]) < tol_outer:
            if verbose:
                print(f"Converged in {iteration + 1} iterations.")
            break

    result = (matrices, vectors)
    if return_errors:
        return result, rec_errors
    return result


# ---------------------------------------------------------------------------
# Scikit-learn-style wrappers
# ---------------------------------------------------------------------------

class LL1(DecompositionMixin):
    r"""LL1 tensor decomposition (unconstrained).

    Represents a 3rd-order tensor ``T`` of shape ``(I, J, K)`` as

    .. math::

        \\mathcal{T} \\approx \\sum_{r=1}^{R} \\mathbf{A}_r \\otimes \\mathbf{c}_r,

    where each :math:`\\mathbf{A}_r` is an ``(I, J)`` activation matrix and
    :math:`\\mathbf{c}_r` is a *K*-vector.  Fitted via unconstrained
    Alternating Least Squares (ALS).

    Parameters
    ----------
    rank : int
        Number of LL1 terms *R*.
    n_iter_max : int, optional
        Maximum number of ALS iterations.  Default ``100``.
    init : ``{'svd', 'random'}`` or 2-tuple, optional
        Factor initialization.  Default ``'svd'``.
    svd : str, optional
        SVD method.  Default ``'truncated_svd'``.
    tol : float, optional
        Convergence tolerance.  Default ``1e-8``.
    random_state : None, int, or ``RandomState``, optional
    verbose : bool, optional

    Attributes
    ----------
    decomposition_ : tuple ``(matrices, vectors)``
        Fitted factors.
    errors_ : list of float
        Per-iteration reconstruction errors.
    """

    def __init__(
        self,
        rank,
        n_iter_max=100,
        init="svd",
        svd="truncated_svd",
        tol=1e-8,
        random_state=None,
        verbose=False,
    ):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, tensor):
        """Decompose an input tensor.

        Parameters
        ----------
        tensor : ndarray of shape ``(I, J, K)``

        Returns
        -------
        (matrices, vectors) : tuple
        """
        result, errors = ll1_als(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            return_errors=True,
        )
        self.decomposition_ = result
        self.errors_ = errors
        return self.decomposition_

    def __repr__(self):
        return f"Rank-{self.rank} LL1 decomposition."


class ConstrainedLL1(LL1):
    r"""LL1 decomposition with Stokes constraints.

    Extends :class:`LL1` to enforce physical constraints arising in Stokes
    polarimetric imaging:

    * **Stokes cone** on each mode-*K* vector :math:`\\mathbf{c}_r`:
      a valid Stokes vector ``[S0, S1, S2, S3]`` must satisfy
      ``S0 >= ||(S1, S2, S3)||`` and ``S0 >= 0``.
    * **Non-negativity** on each activation matrix :math:`\\mathbf{A}_r`.

    Two optimization algorithms are available via the *method* parameter:

    * ``'ao_admm'`` (default): AO-ADMM – update each block via an inner
      ADMM loop, exploiting the mode-*K* unfolding.
    * ``'bpg'``: Block-Proximal Gradient – projected ALS.

    Parameters
    ----------
    rank : int
    n_iter_max : int, optional
        Maximum outer iterations.  Default ``100``.
    n_iter_max_inner : int, optional
        Inner ADMM iterations (only for ``method='ao_admm'``).  Default ``10``.
    init : ``{'svd', 'random'}`` or 2-tuple, optional
    svd : str, optional
    tol : float, optional
        Outer convergence tolerance.  Default ``1e-8``.
    tol_inner : float, optional
        Inner ADMM tolerance (``method='ao_admm'`` only).  Default ``1e-6``.
    method : ``{'ao_admm', 'bpg'}``, optional
        Optimization algorithm.  Default ``'ao_admm'``.
    non_negative_matrices : bool, optional
        Enforce ``A_r >= 0``.  Default ``True``.
    stokes_vectors : bool, optional
        Enforce Stokes cone on each ``c_r``.  Default ``True``.
    random_state : None, int, or ``RandomState``, optional
    verbose : bool, optional

    Attributes
    ----------
    decomposition_ : tuple ``(matrices, vectors)``
    errors_ : list of float
    """

    def __init__(
        self,
        rank,
        n_iter_max=100,
        n_iter_max_inner=10,
        init="svd",
        svd="truncated_svd",
        tol=1e-8,
        tol_inner=1e-6,
        method="ao_admm",
        non_negative_matrices=True,
        stokes_vectors=True,
        random_state=None,
        verbose=False,
    ):
        super().__init__(
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_iter_max_inner = n_iter_max_inner
        self.tol_inner = tol_inner
        self.method = method
        self.non_negative_matrices = non_negative_matrices
        self.stokes_vectors = stokes_vectors

    def fit_transform(self, tensor):
        """Decompose an input tensor with Stokes constraints.

        Parameters
        ----------
        tensor : ndarray of shape ``(I, J, K)``

        Returns
        -------
        (matrices, vectors) : tuple
        """
        if self.method == "ao_admm":
            result, errors = ll1_ao_admm(
                tensor,
                rank=self.rank,
                n_iter_max=self.n_iter_max,
                n_iter_max_inner=self.n_iter_max_inner,
                init=self.init,
                svd=self.svd,
                tol_outer=self.tol,
                tol_inner=self.tol_inner,
                non_negative_matrices=self.non_negative_matrices,
                stokes_vectors=self.stokes_vectors,
                random_state=self.random_state,
                verbose=self.verbose,
                return_errors=True,
            )
        elif self.method == "bpg":
            result, errors = ll1_bpg(
                tensor,
                rank=self.rank,
                n_iter_max=self.n_iter_max,
                init=self.init,
                svd=self.svd,
                tol=self.tol,
                non_negative_matrices=self.non_negative_matrices,
                stokes_vectors=self.stokes_vectors,
                random_state=self.random_state,
                verbose=self.verbose,
                return_errors=True,
            )
        else:
            raise ValueError(
                f'Unknown method "{self.method}". Use "ao_admm" or "bpg".'
            )

        self.decomposition_ = result
        self.errors_ = errors
        return self.decomposition_

    def __repr__(self):
        return f"Rank-{self.rank} Constrained LL1 decomposition (Stokes, {self.method})."
