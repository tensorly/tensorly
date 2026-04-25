"""
Core operations on LL1 tensors.

An LL1 tensor of order three with rank ``R`` and column-rank ``L``
decomposes a tensor ``X`` of shape ``(I, J, K)`` as::

    X[:, :, k] = sum_{r=1}^{R}  C[k, r] * A_r @ B_r^T

where

* ``A`` has shape ``(I, L*R)``.  Columns ``r*L`` to ``(r+1)*L - 1``
  form the matrix ``A_r`` of shape ``(I, L)``.
* ``B`` has shape ``(J, L*R)``.  Columns ``r*L`` to ``(r+1)*L - 1``
  form the matrix ``B_r`` of shape ``(J, L)``.
* ``C`` has shape ``(K, R)``.  Column ``r`` is the vector ``c_r``.

The "LL1" name reflects the ``(L, L, 1)`` rank structure: each term
contributes a rank-``L`` matrix (``A_r @ B_r^T``) in the first two
modes and a rank-1 vector in the third mode.
"""

import numpy as np

from . import backend as T
from ._factorized_tensor import FactorizedTensor


# Author: TensorLy Contributors

# License: BSD 3 clause


def _validate_ll1_tensor(ll1_tensor):
    """Validate an LL1 tensor and return ``(shape, rank, column_rank)``.

    Parameters
    ----------
    ll1_tensor : tuple ``(A, B, C)``
        * ``A`` : ndarray of shape ``(I, L*R)``
        * ``B`` : ndarray of shape ``(J, L*R)``
        * ``C`` : ndarray of shape ``(K, R)``

    Returns
    -------
    shape : tuple ``(I, J, K)``
    rank : int
        The number of LL1 terms ``R``.
    column_rank : int
        The column rank ``L`` of each matrix factor.

    Raises
    ------
    ValueError
        If the dimensions are inconsistent.
    """
    A, B, C = ll1_tensor

    if T.ndim(A) != 2:
        raise ValueError(f"A must be a 2-D matrix, got ndim={T.ndim(A)}.")
    if T.ndim(B) != 2:
        raise ValueError(f"B must be a 2-D matrix, got ndim={T.ndim(B)}.")
    if T.ndim(C) != 2:
        raise ValueError(f"C must be a 2-D matrix, got ndim={T.ndim(C)}.")

    I, LR_A = T.shape(A)
    J, LR_B = T.shape(B)
    K, R = T.shape(C)

    if R == 0:
        raise ValueError("LL1 rank R must be >= 1.")

    if LR_A != LR_B:
        raise ValueError(
            f"A and B must have the same number of columns (L*R), "
            f"got {LR_A} and {LR_B}."
        )

    if LR_A % R != 0:
        raise ValueError(
            f"Number of columns of A ({LR_A}) must be divisible by "
            f"rank R ({R})."
        )

    L = LR_A // R
    if L == 0:
        raise ValueError("Column rank L must be >= 1.")

    shape = (I, J, K)
    return shape, R, L


def ll1_to_tensor(ll1_tensor):
    """Reconstruct a full tensor from an LL1 factorisation.

    An LL1 decomposition with rank ``R`` and column-rank ``L`` is equivalent
    to a CP decomposition with ``L*R`` rank-1 terms where the mode-2 factor
    repeats each column of ``C`` exactly ``L`` times.  Reconstruction is
    delegated to :func:`cp_to_tensor`, which uses a single Khatri–Rao
    product and one matrix multiply instead of a Python loop over ``R``.

    Parameters
    ----------
    ll1_tensor : tuple ``(A, B, C)`` or LL1Tensor

    Returns
    -------
    tensor : ndarray of shape ``(I, J, K)``
    """
    from .cp_tensor import cp_to_tensor

    shape, R, L = _validate_ll1_tensor(ll1_tensor)
    A, B, C = ll1_tensor

    if L == 1:
        # Already a valid CP factorisation; no expansion needed.
        return cp_to_tensor((None, [A, B, C]))

    # Expand C from (K, R) to (K, L*R) so that column r*L+l equals C[:, r].
    # This is C @ expand, where expand is a (R, L*R) matrix with
    # expand[r, r*L:(r+1)*L] = 1.
    expand = T.tensor(np.repeat(np.eye(R), L, axis=1), **T.context(C))
    C_exp = T.dot(C, expand)  # (K, L*R)

    return cp_to_tensor((None, [A, B, C_exp]))


def ll1_to_unfolded(ll1_tensor, mode):
    """Mode-``mode`` unfolding of an LL1 tensor.

    Parameters
    ----------
    ll1_tensor : tuple ``(A, B, C)`` or LL1Tensor
    mode : int

    Returns
    -------
    unfolded : ndarray
    """
    from .base import unfold

    return unfold(ll1_to_tensor(ll1_tensor), mode)


def ll1_to_vec(ll1_tensor):
    """Vectorisation of an LL1 tensor.

    Parameters
    ----------
    ll1_tensor : tuple ``(A, B, C)`` or LL1Tensor

    Returns
    -------
    vec : 1D ndarray
    """
    return T.reshape(ll1_to_tensor(ll1_tensor), (-1,))


class LL1Tensor(FactorizedTensor):
    r"""Factorised tensor in LL1 format.

    An LL1 tensor with rank ``R`` and column-rank ``L`` represents a
    third-order tensor as::

        X[:, :, k] = \sum_{r=1}^{R} C[k, r] \, A_r \, B_r^T

    Parameters
    ----------
    ll1_tensor : tuple ``(A, B, C)``
        * ``A`` : ndarray of shape ``(I, L*R)``
        * ``B`` : ndarray of shape ``(J, L*R)``
        * ``C`` : ndarray of shape ``(K, R)``
    """

    def __init__(self, ll1_tensor):
        super().__init__()

        shape, rank, column_rank = _validate_ll1_tensor(ll1_tensor)
        A, B, C = ll1_tensor

        self.shape = shape
        self.rank = rank
        self.column_rank = column_rank
        self.A = A
        self.B = B
        self.C = C

    def __getitem__(self, index):
        if index == 0:
            return self.A
        elif index == 1:
            return self.B
        elif index == 2:
            return self.C
        else:
            raise IndexError(
                f"You tried to access index {index} of an LL1 tensor.\n"
                "You can only access indices 0, 1, and 2 of an LL1 tensor "
                "(corresponding to A, B, and C respectively)."
            )

    def __setitem__(self, index, value):
        if index == 0:
            self.A = value
        elif index == 1:
            self.B = value
        elif index == 2:
            self.C = value
        else:
            raise IndexError(
                f"You tried to set index {index} of an LL1 tensor.\n"
                "You can only set indices 0, 1, and 2 of an LL1 tensor "
                "(corresponding to A, B, and C respectively)."
            )

    def __iter__(self):
        yield self.A
        yield self.B
        yield self.C

    def __len__(self):
        return 3

    def __repr__(self):
        return (
            f"(A, B, C) : rank-{self.rank} (L={self.column_rank}) "
            f"LL1Tensor of shape {self.shape}"
        )

    def to_tensor(self):
        return ll1_to_tensor(self)

    def to_vec(self):
        return ll1_to_vec(self)

    def to_unfolded(self, mode):
        return ll1_to_unfolded(self, mode)


def check_ll1_uniqueness(shape, rank, column_rank):
    r"""Check generic uniqueness of an LL1 decomposition.

    Domanov and De Lathauwer [1]_ show that the decomposition of a
    third-order tensor of shape ``(I, J, K)`` into ``R`` multilinear
    rank-``(L, L, 1)`` terms is *generically unique* when all terms share
    the same column-rank ``L`` and the following sufficient condition holds::

        R <= min( (I - L)*(J - L),  K )

    The condition is stated in [1]_ for rank-``(1, L, L)`` terms; applying it
    to our rank-``(L, L, 1)`` convention amounts to relabelling the modes so
    that the singleton mode carries dimension ``K``.

    For ``L = 1`` the bound reduces to ``R <= min((I-1)(J-1), K)``, which
    recovers the well-known generic-uniqueness condition for the canonical
    polyadic decomposition (CPD) of third-order tensors.

    Parameters
    ----------
    shape : tuple ``(I, J, K)``
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L`` of each matrix factor.

    Returns
    -------
    unique : bool
        ``True`` when the sufficient condition holds.

    References
    ----------
    .. [1] I. Domanov and L. De Lathauwer, "On uniqueness and computation
           of the decomposition of a tensor into multilinear rank-(1, Lr, Lr)
           terms," SIAM J. Matrix Anal. Appl., vol. 41, no. 2, pp. 713–733,
           2020.
    """
    if len(shape) != 3:
        raise ValueError("Uniqueness check is only defined for third-order tensors.")

    I, J, K = shape
    R = rank
    L = column_rank

    return R <= min((I - L) * (J - L), K)
