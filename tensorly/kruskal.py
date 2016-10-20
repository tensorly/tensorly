import numpy as np
from .base import fold, tensor_to_vec
from .tenalg import khatri_rao

# Author: Jean Kossaifi


def kruskal_to_tensor(factors):
    """Turns the Khatri-product of matrices into a full tensor

        :math:`factor_matrices = [|U_1, ... U_n|]` becomes
        a tensor shape `(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])`

    Parameters
    ----------
    factors : ndarray list
        list of factor matrices, all with the same number of columns
        i.e. for all matrix U in factor_matrices:
        U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

    Returns
    -------
    ndarray
        full tensor of shape (U[1].shape[0], ... U[-1].shape[0])

    Notes
    -----
    This version works by first computing the mode-0 unfolding of the tensor
    and then refolding it.
    There are other possible and equivalent alternate implementation.

    Version slower but closer to the mathematical definition
    of a tensor decomposition:

    >>> from functools import reduce
    >>> def kt_to_tensor(factors):
    ...     for r in range(factors[0].shape[1]):
    ...         vecs = np.ix_(*[u[:, r] for u in factors])
    ...         if r:
    ...             res += reduce(np.multiply, vecs)
    ...         else:
    ...             res = reduce(np.multiply, vecs)
    ...     return res

    """
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
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
    return factors[mode].dot(khatri_rao(factors, skip_matrix=mode).T)


def kruskal_to_vec(factors):
    """Turns the khatri-product of matrices into a vector

        (the tensor ``factors = [|U_1, ... U_n|]``
        is converted into a raveled mode-0 unfolding)

    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        ie for all u in U: u[i] has shape (s_i, R), where R is fixed

    Returns
    -------
    ndarray
        vectorised tensor
    """
    return tensor_to_vec(kruskal_to_tensor(factors))
