from .base import unfold, tensor_to_vec
from .tenalg import multi_mode_dot
from .tenalg import kronecker

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


def tucker_to_tensor(core, factors, skip_factor=None, transpose_factors=False):
    """Converts the Tucker tensor into a full tensor

    Parameters
    ----------
    core : ndarray
       core tensor
    factors : ndarray list
       list of matrices of shape (s_i, core.shape[i])
    skip_factor : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of `tensor.ndim`
    transpose_factors : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    2D-array
       full tensor of shape `(U[0].shape[0], ..., U[-1].shape[0])`

    Notes
    -----
    This implementation is equivalent to:

    >>> def tucker_to_tensor(G, U):
    ...     for i, matrix in enumerate(U):
    ...         if not i:
    ...             res = mode_dot(G, matrix, i)
    ...         else:
    ...             res = mode_dot(res, matrix, i)
    ...     return res
    """
    return multi_mode_dot(core, factors, skip=skip_factor, transpose=transpose_factors)


def tucker_to_unfolded(core, factors, mode=0, skip_factor=None, transpose_factors=False):
    """Converts the Tucker decomposition into an unfolded tensor (i.e. a matrix)

    Parameters
    ----------
    G : ndarray
        core tensor
    U : ndarray list
        list of matrices
    mode : None or int list, optional, default is None
    skip_factor : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of `tensor.ndim`
    transpose_factors : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    2D-array
        unfolded tensor
    """
    return unfold(tucker_to_tensor(core, factors, skip_factor=skip_factor, transpose_factors=transpose_factors), mode)


def tucker_to_vec(core, factors, skip_factor=None, transpose_factors=False):
    """Converts a Tucker decomposition into a vectorised tensor

    Parameters
    ----------
    core : ndarray
        core tensor
    factors : ndarray list
        list of factor matrices
    skip_factor : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of `tensor.ndim`
    transpose_factors : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    1D-array
        vectorised tensor

    Notes
    -----
    Mathematically equivalent but much slower,
    you can obtain the same result using:

    >>> def tucker_to_vec(core, factors):
    ...     return kronecker(factors).dot(tensor_to_vec(core))

    In this implementation we:

    1* take the n-mode product of the core with all the factors
    2* vectorize the result
       (rather than computing potentially large kronecker product of factors)
    """
    return tensor_to_vec(tucker_to_tensor(core, factors, skip_factor=skip_factor, transpose_factors=transpose_factors))

