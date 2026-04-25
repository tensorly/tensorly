"""
LL1 decomposition via Alternating Least Squares.

An LL1 decomposition expresses a third-order tensor ``X`` of shape
``(I, J, K)`` as::

    X[:, :, k] = sum_{r=1}^{R}  C[k, r] * A_r @ B_r^T

where ``A_r`` is ``(I, L)``, ``B_r`` is ``(J, L)``, and ``C[:, r]``
is ``(K,)``.  The full factor matrices are ``A`` of shape ``(I, L*R)``,
``B`` of shape ``(J, L*R)``, and ``C`` of shape ``(K, R)``.
"""

import numpy as np

import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..ll1_tensor import LL1Tensor, ll1_to_tensor


# Author: TensorLy Contributors

# License: BSD 3 clause


# ---------------------------------------------------------------------------
# Private helpers used by ll1_als
# ---------------------------------------------------------------------------

_REG = 1e-12  # Tikhonov regularisation added to every Gram matrix


def _block_expand(small, R, L):
    """Tile each entry of an ``(R, R)`` array into an ``L x L`` block.

    The result is ``(L*R, L*R)``.  Used to expand ``C^T C`` so it can be
    element-wise multiplied with the full ``(L*R, L*R)`` Gram of ``A`` or
    ``B``.
    """
    return np.repeat(np.repeat(small, L, axis=0), L, axis=1)


def _block_sum(large, R, L):
    """Sum the ``L x L`` sub-blocks of an ``(L*R, L*R)`` array.

    Returns an ``(R, R)`` array where entry ``(r1, r2)`` is the sum over the
    ``L x L`` block at rows ``r1*L:(r1+1)*L``, columns ``r2*L:(r2+1)*L``.
    Used to build the ``(R, R)`` Gram matrix for the ``C`` update.
    """
    return large.reshape(R, L, R, L).sum(axis=(1, 3))


def _solve_nrm(gram, rhs):
    r"""Solve the symmetric normal equations ``X @ gram = rhs``.

    Computed as ``X = (gram \\ rhs^T)^T`` via :func:`tensorly.solve`.

    Parameters
    ----------
    gram : (S, S) tensor – symmetric positive (semi-)definite
    rhs  : (N, S) tensor

    Returns
    -------
    X : (N, S) tensor satisfying ``X @ gram = rhs``
    """
    return tl.transpose(tl.solve(gram, tl.transpose(rhs)))


def initialize_ll1(tensor, rank, column_rank, init="random", random_state=None):
    r"""Initialize factors ``(A, B, C)`` for an LL1 decomposition.

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, K)``
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L`` of each matrix factor.
    init : {'random', LL1Tensor}, optional
        Initialisation strategy.
    random_state : {None, int, np.random.RandomState}

    Returns
    -------
    A : ndarray of shape ``(I, L*R)``
    B : ndarray of shape ``(J, L*R)``
    C : ndarray of shape ``(K, R)``
    """
    rng = tl.check_random_state(random_state)
    I, J, K = tl.shape(tensor)
    R = rank
    L = column_rank

    if init == "random":
        A = tl.tensor(rng.random_sample((I, L * R)), **tl.context(tensor))
        B = tl.tensor(rng.random_sample((J, L * R)), **tl.context(tensor))
        C = tl.tensor(rng.random_sample((K, R)), **tl.context(tensor))
    elif isinstance(init, (tuple, list, LL1Tensor)):
        try:
            ll1 = LL1Tensor(init)
            A, B, C = ll1.A, ll1.B, ll1.C
        except ValueError:
            raise ValueError(
                "If initialization is a mapping, it must be convertible "
                "to an LL1Tensor."
            )
    else:
        raise ValueError(f'Initialization method "{init}" not recognized.')

    return A, B, C


def ll1_als(
    tensor,
    rank,
    column_rank,
    n_iter_max=100,
    init="random",
    tol=1e-8,
    random_state=None,
    verbose=0,
    return_errors=False,
):
    r"""LL1 decomposition via Alternating Least Squares (ALS).

    Computes a rank-``R``, column-rank-``L`` LL1 decomposition::

        X[:, :, k] \approx \sum_{r=1}^{R} C[k, r]\, A_r\, B_r^T

    All three factor matrices ``A``, ``B``, ``C`` are updated in closed
    form at every iteration using the mode unfoldings of the tensor and
    the corresponding Gram matrices.

    Parameters
    ----------
    tensor : ndarray of shape ``(I, J, K)``
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L`` of each ``(I, L)`` and ``(J, L)`` block.
    n_iter_max : int, optional
        Maximum number of ALS iterations.
    init : {'random', LL1Tensor}, optional
        Initialisation strategy.
    tol : float, optional
        Convergence tolerance on relative reconstruction error change.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity.
    return_errors : bool, optional
        If ``True``, also return the list of per-iteration errors.

    Returns
    -------
    LL1Tensor : (A, B, C)
        * ``A`` : ndarray of shape ``(I, L*R)``
        * ``B`` : ndarray of shape ``(J, L*R)``
        * ``C`` : ndarray of shape ``(K, R)``
    errors : list of float
        Reconstruction errors (only if ``return_errors=True``).

    References
    ----------
    .. [1] K. B. Kolda and B. W. Bader, "Tensor Decompositions and
           Applications", SIAM Review, vol. 51, no. 3, pp. 455-500, 2009.
    """
    if tl.ndim(tensor) != 3:
        raise ValueError("LL1 decomposition requires a third-order tensor.")

    I, J, K = tl.shape(tensor)
    R = rank
    L = column_rank

    A, B, C = initialize_ll1(
        tensor, rank, column_rank, init=init, random_state=random_state
    )

    # Tikhonov regularisation matrices – constant across iterations
    reg_LR = _REG * np.eye(L * R)
    reg_R = _REG * np.eye(R)

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):
        # ----- shared contraction with C -----
        # T_xc[i, j, r] = sum_k X[i, j, k] * C[k, r]
        # Reused by both the A and B updates.
        T_xc = tl.tensordot(tensor, C, axes=([2], [0]))  # (I, J, R)

        # C^T C is the same for A and B this iteration; expand it once
        # into an (LR, LR) array so it can be element-wise multiplied
        # with the (LR, LR) Gram of the other factor.
        gram_C_blk = _block_expand(
            tl.to_numpy(tl.dot(tl.transpose(C), C)), R, L
        )  # (LR, LR)

        # ----- update A -----
        # Normal equations:  A @ G_A = M_A
        #   M_A[:, r*L:(r+1)*L] = T_xc[:,:,r] @ B_r
        #   G_A = gram_C_blk  *  (B^T B)          [element-wise]
        M_A = tl.concatenate(
            [tl.dot(T_xc[:, :, r], B[:, r * L : (r + 1) * L]) for r in range(R)],
            axis=1,
        )  # (I, LR)
        gram_B = tl.to_numpy(tl.dot(tl.transpose(B), B))  # (LR, LR)
        G_A = tl.tensor(gram_C_blk * gram_B + reg_LR, **tl.context(tensor))
        A = _solve_nrm(G_A, M_A)

        # ----- update B -----
        # Normal equations:  B @ G_B = M_B
        #   M_B[:, r*L:(r+1)*L] = T_xc[:,:,r]^T @ A_r
        #   G_B = gram_C_blk  *  (A^T A)
        M_B = tl.concatenate(
            [
                tl.dot(tl.transpose(T_xc[:, :, r]), A[:, r * L : (r + 1) * L])
                for r in range(R)
            ],
            axis=1,
        )  # (J, LR)
        gram_A = tl.to_numpy(tl.dot(tl.transpose(A), A))  # (LR, LR)
        G_B = tl.tensor(gram_C_blk * gram_A + reg_LR, **tl.context(tensor))
        B = _solve_nrm(G_B, M_B)

        # ----- update C -----
        # Normal equations:  C @ G_M = M_C
        #   M_C[:, r]     = sum_{i,j} X[i,j,:] * (A_r @ B_r^T)[i,j]
        #   G_M[r1, r2]   = block_sum( (A^T A) * (B^T B) )
        M_C_cols = []
        for r in range(R):
            A_r = A[:, r * L : (r + 1) * L]
            B_r = B[:, r * L : (r + 1) * L]
            M_r = tl.dot(A_r, tl.transpose(B_r))  # (I, J)
            M_C_cols.append(
                tl.reshape(
                    tl.tensordot(tensor, M_r, axes=([0, 1], [0, 1])), (K, 1)
                )
            )
        M_C = tl.concatenate(M_C_cols, axis=1)  # (K, R)

        gram_B = tl.to_numpy(tl.dot(tl.transpose(B), B))  # recompute: B changed
        G_M = tl.tensor(
            _block_sum(gram_A * gram_B, R, L) + reg_R, **tl.context(tensor)
        )
        C = _solve_nrm(G_M, M_C)

        # ----- convergence -----
        rec_error = tl.norm(tensor - ll1_to_tensor((A, B, C)), 2) / norm_tensor
        rec_errors.append(rec_error)

        if verbose:
            print(f"iteration {iteration}, reconstruction error: {rec_error}")

        if tol and iteration >= 1:
            delta = float(tl.to_numpy(rec_errors[-2])) - float(
                tl.to_numpy(rec_errors[-1])
            )
            if abs(delta) < tol:
                if verbose:
                    print(f"LL1 ALS converged after {iteration} iterations.")
                break

    ll1_tensor = LL1Tensor((A, B, C))

    if return_errors:
        return ll1_tensor, rec_errors
    return ll1_tensor


class LL1(DecompositionMixin):
    r"""LL1 decomposition via Alternating Least Squares.

    Decomposes a third-order tensor ``X`` of shape ``(I, J, K)`` as::

        X[:, :, k] \approx \sum_{r=1}^{R} C[k, r]\, A_r\, B_r^T

    Parameters
    ----------
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L`` of each matrix factor block.
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
        The fitted decomposition.
    errors_ : list of float
        Reconstruction errors per iteration.

    References
    ----------
    .. [1] K. B. Kolda and B. W. Bader, "Tensor Decompositions and
           Applications", SIAM Review, vol. 51, no. 3, pp. 455-500, 2009.
    """

    def __init__(
        self,
        rank,
        column_rank,
        n_iter_max=100,
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
        tensor : ndarray of shape ``(I, J, K)``

        Returns
        -------
        LL1Tensor
            The decomposed tensor.
        """
        ll1_tensor, errors = ll1_als(
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
