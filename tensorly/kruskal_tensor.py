"""
Core operations on Kruskal tensors.
"""

from . import backend as T
from .base import fold, tensor_to_vec
from .tenalg import khatri_rao

# Author: Jean Kossaifi

# License: BSD 3 clause


def kruskal_to_tensor(factors):
    """Turns the Khatri-product of matrices into a full tensor

        ``factor_matrices = [|U_1, ... U_n|]`` becomes
        a tensor shape ``(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])``

    Parameters
    ----------
    factors : ndarray list
        list of factor matrices, all with the same number of columns
        i.e. for all matrix U in factor_matrices:
        U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

    Returns
    -------
    ndarray
        full tensor of shape ``(U[1].shape[0], ... U[-1].shape[0])``

    Notes
    -----
    This version works by first computing the mode-0 unfolding of the tensor
    and then refolding it.

    There are other possible and equivalent alternate implementation, e.g.
    summing over r and updating an outer product of vectors.
    """
    shape = [factor.shape[0] for factor in factors]
    full_tensor = T.dot(factors[0], T.transpose(khatri_rao(factors[1:])))
    return fold(full_tensor, 0, shape)


def kruskal_to_unfolded(factors, mode):
    """Turns the khatri-product of matrices into an unfolded tensor

        turns ``factors = [|U_1, ... U_n|]`` into a mode-`mode`
        unfolding of the tensor

    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        ie for all u in factor_matrices:
        u[i] has shape (s_u_i, R), where R is fixed
    mode: int
        mode of the desired unfolding

    Returns
    -------
    ndarray
        unfolded tensor of shape (tensor_shape[mode], -1)

    Notes
    -----
    Writing factors = [U_1, ..., U_n], we exploit the fact that
    ``U_k = U[k].dot(khatri_rao(U_1, ..., U_k-1, U_k+1, ..., U_n))``
    """
    return T.dot(factors[mode], T.transpose(khatri_rao(factors, skip_matrix=mode)))


def kruskal_to_vec(factors):
    """Turns the khatri-product of matrices into a vector

        (the tensor ``factors = [|U_1, ... U_n|]``
        is converted into a raveled mode-0 unfolding)

    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        i.e.::

            for u in U:
                u[i].shape == (s_i, R)
                
        where `R` is fixed while `s_i` can vary with `i`

    Returns
    -------
    ndarray
        vectorised tensor
    """
    return tensor_to_vec(kruskal_to_tensor(factors))

def normalize_kruskal(factors):
    """Normalizes factors to unit length and returns factor magnitudes
        turns  ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,
        where the columns of each `V_k` are normalized to unit Euclidean length
        from the columns of `U_k` with the normalizing constants absorbed
        into `weights`. In the special case of a symmetric tensor, `weights`
        holds the eigenvalues of the tensor.

    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        i.e.::
            for u in U:
                u[i].shape == (s_i, R)
                
        where `R` is fixed while `s_i` can vary with `i`
    Returns
    -------
    normalized_factors : ndarray list
        list of matrices with the same shape as `factors`
    weights : ndarray
        vector of length `R` holding normalizing constants
    """

    # allocate variables for weights, and normalized factors
    rank = factors[0].shape[0]
    weights = T.ones(rank)
    V = []

    # normalize columns of factor matrices
    for factor in factors:
        s = T.norm(factor, axis=0)
        weights *= s
        V.append(factor / T.where(s==0, 1, s))

    return V, weights
