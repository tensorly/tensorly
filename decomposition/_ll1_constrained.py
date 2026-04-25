"""
Stokes-constrained LL1 decomposition.

For Stokes imaging, each column of ``C`` (the vector factor in mode 2)
must be a valid Stokes vector ``[s0, s1, s2, s3]`` satisfying::

    s0 >= 0   and   s0^2 >= s1^2 + s2^2 + s3^2

and the columns of ``A`` and ``B`` must be non-negative so that
``A_r @ B_r^T`` forms a valid non-negative activation map.

Two optimisation algorithms are provided:

* **Block Proximal Gradient (BPG)** – each block ``A_r``, ``B_r``,
  ``c_r`` is updated via an exact least-squares step followed by the
  appropriate constraint projection.
* **AO-ADMM** – Alternating Optimisation with ADMM; each block update
  is wrapped in a short ADMM loop whose ``x``-step solves a
  ``(L x L)`` or scalar linear system derived from the mode unfolding.
"""

import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ._ll1 import initialize_ll1
from ..ll1_tensor import LL1Tensor, ll1_to_tensor


# Author: TensorLy Contributors

# License: BSD 3 clause


def _project_stokes(s):
    """Project a single length-4 vector onto the Stokes cone.

    The Stokes cone is::

        { s in R^4 : s[0] >= 0,  s[0]^2 >= s[1]^2 + s[2]^2 + s[3]^2 }

    Parameters
    ----------
    s : ndarray of shape ``(4,)``

    Returns
    -------
    s_proj : ndarray of shape ``(4,)``
    """
    s0 = s[0]
    s_rest = s[1:]
    polaris = tl.sqrt(tl.sum(s_rest * s_rest))

    inside = (s0 >= 0) & (s0 * s0 >= polaris * polaris)
    neg_ray = (s0 < 0) & (s0 * s0 >= polaris * polaris)

    s0_new = (s0 + polaris) / 2.0
    scale = tl.where(polaris > 0, s0_new / polaris, tl.zeros_like(polaris))
    s_rest_new = s_rest * tl.reshape(scale, (1,))

    proj_s0 = tl.where(inside, s0, tl.where(neg_ray, tl.zeros_like(s0), s0_new))
    proj_rest = tl.where(
        tl.reshape(inside, (1,)),
        s_rest,
        tl.where(tl.reshape(neg_ray, (1,)), tl.zeros_like(s_rest), s_rest_new),
    )

    return tl.concatenate([tl.reshape(proj_s0, (1,)), proj_rest], axis=0)


def ll1_bpg(
    tensor,
    rank,
    column_rank,
    n_iter_max=200,
    init="random",
    tol=1e-8,
    random_state=None,
    verbose=0,
    return_errors=False,
):
    r"""Stokes-constrained LL1 via Block Proximal Gradient.

    Each block ``A_r`` and ``B_r`` is constrained to be **non-negative**
    and each column ``c_r`` of ``C`` is projected onto the **Stokes
    cone**.  At every iteration the exact unconstrained least-squares
    solution is computed for each block and then projected.

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, 4)``
        The third mode must have size 4.
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L``.
    n_iter_max : int, optional
        Maximum number of iterations.
    init : {'random', LL1Tensor}, optional
        Initialisation strategy.
    tol : float, optional
        Convergence tolerance.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity.
    return_errors : bool, optional
        If ``True``, return the list of reconstruction errors.

    Returns
    -------
    LL1Tensor : (A, B, C)
    errors : list of float
        Only returned when ``return_errors=True``.

    References
    ----------
    .. [1] N. Absil, R. Mahony, R. Sepulchre, "Optimization Algorithms on
           Matrix Manifolds", Princeton University Press, 2008.
    """
    if tl.ndim(tensor) != 3:
        raise ValueError("LL1 BPG requires a third-order tensor.")
    if tl.shape(tensor)[2] != 4:
        raise ValueError(
            "Stokes-constrained LL1 requires the third mode to have size 4."
        )

    I, J, K = tl.shape(tensor)
    R = rank
    L = column_rank

    A, B, C = initialize_ll1(
        tensor, rank, column_rank, init=init, random_state=random_state
    )
    # Enforce constraints on initialisation
    A = tl.clip(A, 0)
    B = tl.clip(B, 0)
    # Project each column of C onto Stokes cone
    cols_C = []
    for r in range(R):
        cols_C.append(tl.reshape(_project_stokes(C[:, r]), (-1, 1)))
    C = tl.concatenate(cols_C, axis=1)

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):
        # ---------------------------------------------------------
        # Update each A_r block  (I x L), then project >= 0
        # ---------------------------------------------------------
        for r in range(R):
            # Residual without term r
            residual = tensor
            for s in range(R):
                if s != r:
                    A_s = A[:, s * L : (s + 1) * L]
                    B_s = B[:, s * L : (s + 1) * L]
                    c_s = C[:, s]
                    M_s = tl.dot(A_s, tl.transpose(B_s))
                    residual = residual - tl.reshape(M_s, (I, J, 1)) * tl.reshape(
                        c_s, (1, 1, K)
                    )

            c_r = C[:, r]  # (K,)
            B_r = B[:, r * L : (r + 1) * L]  # (J, L)

            # T_R[i, j] = sum_k residual[i,j,k] * c_r[k]
            T_R = tl.tensordot(residual, c_r, axes=([2], [0]))  # (I, J)

            # Normal equations: A_r @ (||c_r||^2 * B_r^T B_r) = T_R @ B_r
            c_sq = tl.sum(c_r * c_r)
            H_r = c_sq * tl.dot(tl.transpose(B_r), B_r)  # (L, L)

            rhs = tl.dot(T_R, B_r)  # (I, L)
            # A_r = rhs @ H_r^{-1}
            A_r_new = tl.transpose(tl.solve(H_r, tl.transpose(rhs)))
            # Project onto non-negative orthant
            A_r_new = tl.clip(A_r_new, 0)

            # Write block back using concatenation
            A = tl.concatenate(
                [A[:, : r * L], A_r_new, A[:, (r + 1) * L :]], axis=1
            )

        # ---------------------------------------------------------
        # Update each B_r block  (J x L), then project >= 0
        # ---------------------------------------------------------
        for r in range(R):
            residual = tensor
            for s in range(R):
                if s != r:
                    A_s = A[:, s * L : (s + 1) * L]
                    B_s = B[:, s * L : (s + 1) * L]
                    c_s = C[:, s]
                    M_s = tl.dot(A_s, tl.transpose(B_s))
                    residual = residual - tl.reshape(M_s, (I, J, 1)) * tl.reshape(
                        c_s, (1, 1, K)
                    )

            c_r = C[:, r]
            A_r = A[:, r * L : (r + 1) * L]  # (I, L)

            T_R = tl.tensordot(residual, c_r, axes=([2], [0]))  # (I, J)

            c_sq = tl.sum(c_r * c_r)
            H_r = c_sq * tl.dot(tl.transpose(A_r), A_r)  # (L, L)

            rhs = tl.dot(tl.transpose(T_R), A_r)  # (J, L)
            B_r_new = tl.transpose(tl.solve(H_r, tl.transpose(rhs)))
            B_r_new = tl.clip(B_r_new, 0)

            B = tl.concatenate(
                [B[:, : r * L], B_r_new, B[:, (r + 1) * L :]], axis=1
            )

        # ---------------------------------------------------------
        # Update each c_r column  (K,), then project onto Stokes
        # ---------------------------------------------------------
        cols_C = []
        for r in range(R):
            residual = tensor
            for s in range(R):
                if s != r:
                    A_s = A[:, s * L : (s + 1) * L]
                    B_s = B[:, s * L : (s + 1) * L]
                    c_s = C[:, s]
                    M_s = tl.dot(A_s, tl.transpose(B_s))
                    residual = residual - tl.reshape(M_s, (I, J, 1)) * tl.reshape(
                        c_s, (1, 1, K)
                    )

            A_r = A[:, r * L : (r + 1) * L]
            B_r = B[:, r * L : (r + 1) * L]
            M_r = tl.dot(A_r, tl.transpose(B_r))  # (I, J)
            M_sq = tl.sum(M_r * M_r)

            # c_r[k] = <residual[:,:,k], M_r> / ||M_r||_F^2
            c_r_new = tl.tensordot(residual, M_r, axes=([0, 1], [0, 1])) / M_sq
            c_r_new = _project_stokes(c_r_new)
            cols_C.append(tl.reshape(c_r_new, (-1, 1)))
        C = tl.concatenate(cols_C, axis=1)

        # Reconstruction error
        rec_error = tl.norm(tensor - ll1_to_tensor((A, B, C)), 2) / norm_tensor
        rec_errors.append(rec_error)

        if verbose:
            print(f"iteration {iteration}, reconstruction error: {rec_error}")

        if tol and iteration >= 1:
            if tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print(f"LL1 BPG converged after {iteration} iterations.")
                break

    ll1_tensor = LL1Tensor((A, B, C))
    if return_errors:
        return ll1_tensor, rec_errors
    return ll1_tensor


def ll1_ao_admm(
    tensor,
    rank,
    column_rank,
    n_iter_max=200,
    n_admm_iter=10,
    rho=1.0,
    init="random",
    tol=1e-8,
    random_state=None,
    verbose=0,
    return_errors=False,
):
    r"""Stokes-constrained LL1 via AO-ADMM.

    Alternating Optimisation with ADMM.  For each block ``A_r``,
    ``B_r``, or ``c_r`` a short ADMM loop is executed.  The
    unconstrained ``x``-step solves an ``(L x L)`` system (for matrix
    blocks) or a scalar equation (for vector blocks) derived from the
    mode unfolding.  The ``z``-step applies the constraint projection.

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, 4)``
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L``.
    n_iter_max : int, optional
        Maximum number of outer AO iterations.
    n_admm_iter : int, optional
        Number of inner ADMM iterations per block.
    rho : float, optional
        ADMM penalty parameter.
    init : {'random', LL1Tensor}, optional
        Initialisation strategy.
    tol : float, optional
        Convergence tolerance.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity.
    return_errors : bool, optional
        If ``True``, return reconstruction errors.

    Returns
    -------
    LL1Tensor : (A, B, C)
    errors : list of float
        Only returned when ``return_errors=True``.

    References
    ----------
    .. [1] S. Boyd, N. Parikh, E. J. Candès, B. Recht, J. Romberg,
           "Distributed Optimization and Statistical Learning via the
           Method of Multipliers", IEEE Trans. Syst., Man, Cybernetics, 2011.
    """
    if tl.ndim(tensor) != 3:
        raise ValueError("LL1 AO-ADMM requires a third-order tensor.")
    if tl.shape(tensor)[2] != 4:
        raise ValueError(
            "Stokes-constrained LL1 requires the third mode to have size 4."
        )

    I, J, K = tl.shape(tensor)
    R = rank
    L = column_rank

    A, B, C = initialize_ll1(
        tensor, rank, column_rank, init=init, random_state=random_state
    )
    A = tl.clip(A, 0)
    B = tl.clip(B, 0)
    cols_C = []
    for r in range(R):
        cols_C.append(tl.reshape(_project_stokes(C[:, r]), (-1, 1)))
    C = tl.concatenate(cols_C, axis=1)

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    rho_eye_L = rho * tl.eye(L, **tl.context(tensor))

    for iteration in range(n_iter_max):
        # ---------------------------------------------------------
        # Update each A_r via ADMM  (I x L)
        # ---------------------------------------------------------
        for r in range(R):
            residual = tensor
            for s in range(R):
                if s != r:
                    A_s = A[:, s * L : (s + 1) * L]
                    B_s = B[:, s * L : (s + 1) * L]
                    c_s = C[:, s]
                    M_s = tl.dot(A_s, tl.transpose(B_s))
                    residual = residual - tl.reshape(M_s, (I, J, 1)) * tl.reshape(
                        c_s, (1, 1, K)
                    )

            c_r = C[:, r]
            B_r = B[:, r * L : (r + 1) * L]

            T_R = tl.tensordot(residual, c_r, axes=([2], [0]))  # (I, J)
            c_sq = tl.sum(c_r * c_r)
            H_r = c_sq * tl.dot(tl.transpose(B_r), B_r)  # (L, L)
            rhs_ls = tl.dot(T_R, B_r)  # (I, L)  — unconstrained target direction

            # ADMM variables
            A_r = A[:, r * L : (r + 1) * L]
            Z_r = tl.copy(A_r)
            U_r = tl.zeros_like(A_r)

            # LHS matrix for x-step: (H_r + rho*I_L)
            lhs = H_r + rho_eye_L  # (L, L)

            for _ in range(n_admm_iter):
                # x-step: A_r = (rhs_ls + rho*(Z_r - U_r)) @ inv(H_r + rho*I)
                rhs = rhs_ls + rho * (Z_r - U_r)  # (I, L)
                A_r = tl.transpose(tl.solve(lhs, tl.transpose(rhs)))
                # z-step: project
                Z_r = tl.clip(A_r + U_r, 0)
                # u-step
                U_r = U_r + A_r - Z_r

            A = tl.concatenate(
                [A[:, : r * L], Z_r, A[:, (r + 1) * L :]], axis=1
            )

        # ---------------------------------------------------------
        # Update each B_r via ADMM  (J x L)
        # ---------------------------------------------------------
        for r in range(R):
            residual = tensor
            for s in range(R):
                if s != r:
                    A_s = A[:, s * L : (s + 1) * L]
                    B_s = B[:, s * L : (s + 1) * L]
                    c_s = C[:, s]
                    M_s = tl.dot(A_s, tl.transpose(B_s))
                    residual = residual - tl.reshape(M_s, (I, J, 1)) * tl.reshape(
                        c_s, (1, 1, K)
                    )

            c_r = C[:, r]
            A_r = A[:, r * L : (r + 1) * L]

            T_R = tl.tensordot(residual, c_r, axes=([2], [0]))
            c_sq = tl.sum(c_r * c_r)
            H_r = c_sq * tl.dot(tl.transpose(A_r), A_r)  # (L, L)
            rhs_ls = tl.dot(tl.transpose(T_R), A_r)  # (J, L)

            B_r = B[:, r * L : (r + 1) * L]
            Z_r = tl.copy(B_r)
            U_r = tl.zeros_like(B_r)

            lhs = H_r + rho_eye_L

            for _ in range(n_admm_iter):
                rhs = rhs_ls + rho * (Z_r - U_r)
                B_r = tl.transpose(tl.solve(lhs, tl.transpose(rhs)))
                Z_r = tl.clip(B_r + U_r, 0)
                U_r = U_r + B_r - Z_r

            B = tl.concatenate(
                [B[:, : r * L], Z_r, B[:, (r + 1) * L :]], axis=1
            )

        # ---------------------------------------------------------
        # Update each c_r via ADMM  (K,)
        # ---------------------------------------------------------
        cols_C = []
        for r in range(R):
            residual = tensor
            for s in range(R):
                if s != r:
                    A_s = A[:, s * L : (s + 1) * L]
                    B_s = B[:, s * L : (s + 1) * L]
                    c_s = C[:, s]
                    M_s = tl.dot(A_s, tl.transpose(B_s))
                    residual = residual - tl.reshape(M_s, (I, J, 1)) * tl.reshape(
                        c_s, (1, 1, K)
                    )

            A_r = A[:, r * L : (r + 1) * L]
            B_r = B[:, r * L : (r + 1) * L]
            M_r = tl.dot(A_r, tl.transpose(B_r))
            M_sq = tl.sum(M_r * M_r)  # scalar Hessian

            # Unconstrained target
            target = tl.tensordot(residual, M_r, axes=([0, 1], [0, 1])) / M_sq  # (K,)

            c_r = C[:, r]
            Z_r = tl.copy(c_r)
            U_r = tl.zeros_like(c_r)

            for _ in range(n_admm_iter):
                # x-step: scalar system  (M_sq + rho) * c = M_sq*target + rho*(Z - U)
                c_r = (M_sq * target + rho * (Z_r - U_r)) / (M_sq + rho)
                # z-step
                Z_r = _project_stokes(c_r + U_r)
                # u-step
                U_r = U_r + c_r - Z_r

            cols_C.append(tl.reshape(Z_r, (-1, 1)))
        C = tl.concatenate(cols_C, axis=1)

        # Reconstruction error
        rec_error = tl.norm(tensor - ll1_to_tensor((A, B, C)), 2) / norm_tensor
        rec_errors.append(rec_error)

        if verbose:
            print(f"iteration {iteration}, reconstruction error: {rec_error}")

        if tol and iteration >= 1:
            if tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print(f"LL1 AO-ADMM converged after {iteration} iterations.")
                break

    ll1_tensor = LL1Tensor((A, B, C))
    if return_errors:
        return ll1_tensor, rec_errors
    return ll1_tensor


class LL1_BPG(DecompositionMixin):
    r"""Stokes-constrained LL1 via Block Proximal Gradient.

    Parameters
    ----------
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L``.
    n_iter_max : int, optional
        Maximum number of iterations.
    init : {'random', LL1Tensor}, optional
        Initialisation strategy.
    tol : float, optional
        Convergence tolerance.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity.

    Attributes
    ----------
    decomposition_ : LL1Tensor
    errors_ : list of float
    """

    def __init__(
        self,
        rank,
        column_rank,
        n_iter_max=200,
        init="random",
        tol=1e-8,
        random_state=None,
        verbose=0,
    ):
        self.rank = rank
        self.column_rank = column_rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, tensor):
        """Decompose an input tensor.

        Parameters
        ----------
        tensor : ndarray of shape ``(I, J, 4)``

        Returns
        -------
        LL1Tensor
        """
        ll1_tensor, errors = ll1_bpg(
            tensor,
            rank=self.rank,
            column_rank=self.column_rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            return_errors=True,
        )
        self.decomposition_ = ll1_tensor
        self.errors_ = errors
        return self.decomposition_


class LL1_AO_ADMM(DecompositionMixin):
    r"""Stokes-constrained LL1 via AO-ADMM.

    Parameters
    ----------
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L``.
    n_iter_max : int, optional
        Maximum number of outer iterations.
    n_admm_iter : int, optional
        Number of inner ADMM iterations per block.
    rho : float, optional
        ADMM penalty parameter.
    init : {'random', LL1Tensor}, optional
        Initialisation strategy.
    tol : float, optional
        Convergence tolerance.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity.

    Attributes
    ----------
    decomposition_ : LL1Tensor
    errors_ : list of float
    """

    def __init__(
        self,
        rank,
        column_rank,
        n_iter_max=200,
        n_admm_iter=10,
        rho=1.0,
        init="random",
        tol=1e-8,
        random_state=None,
        verbose=0,
    ):
        self.rank = rank
        self.column_rank = column_rank
        self.n_iter_max = n_iter_max
        self.n_admm_iter = n_admm_iter
        self.rho = rho
        self.init = init
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, tensor):
        """Decompose an input tensor.

        Parameters
        ----------
        tensor : ndarray of shape ``(I, J, 4)``

        Returns
        -------
        LL1Tensor
        """
        ll1_tensor, errors = ll1_ao_admm(
            tensor,
            rank=self.rank,
            column_rank=self.column_rank,
            n_iter_max=self.n_iter_max,
            n_admm_iter=self.n_admm_iter,
            rho=self.rho,
            init=self.init,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            return_errors=True,
        )
        self.decomposition_ = ll1_tensor
        self.errors_ = errors
        return self.decomposition_
