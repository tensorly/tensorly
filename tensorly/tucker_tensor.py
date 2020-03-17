"""
Core operations on Tucker tensors.
"""

from .base import unfold, tensor_to_vec
from .tenalg import multi_mode_dot, mode_dot
from .tenalg import kronecker
from . import backend as tl
import warnings

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>

# License: BSD 3 clause

def _validate_tucker_tensor(tucker_tensor):
    core, factors = tucker_tensor
    
    if len(factors) < 2:
        raise ValueError('A Tucker tensor should be composed of at least two factors and a core.'
                         'However, {} factor was given.'.format(len(factors)))

    if len(factors) != tl.ndim(core):
        raise ValueError('Tucker decompositions should have one factor per more of the core tensor.'
                         'However, core has {} modes but {} factors have been provided'.format(
                         tl.ndim(core), len(factors)))

    shape = []
    rank = []
    for i, factor in enumerate(factors):
        current_shape, current_rank = tl.shape(factor)
        if current_rank != tl.shape(core)[i]:
            raise ValueError('Factor `n` of Tucker decomposition should verify:\n'
                             'factors[n].shape[1] = core.shape[n].'
                             'However, factors[{0}].shape[1]={1} but core.shape[{0}]={2}.'.format(
                                 i, tl.shape(factor)[1], tl.shape(core)[i]))
        shape.append(current_shape)
        rank.append(current_rank)

    return tuple(shape), tuple(rank)

def tucker_to_tensor(tucker_tensor, skip_factor=None, transpose_factors=False):
    """Converts the Tucker tensor into a full tensor

    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices
    skip_factor : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of ``tensor.ndim``
    transpose_factors : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    2D-array
       full tensor of shape ``(factors[0].shape[0], ..., factors[-1].shape[0])``
    """
    core, factors = tucker_tensor
    return multi_mode_dot(core, factors, skip=skip_factor, transpose=transpose_factors)


def tucker_to_unfolded(tucker_tensor, mode=0, skip_factor=None, transpose_factors=False):
    """Converts the Tucker decomposition into an unfolded tensor (i.e. a matrix)

    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices
    mode : None or int list, optional, default is None
    skip_factor : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a length of ``tensor.ndim``
    transpose_factors : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    2D-array
        unfolded tensor
    """
    return unfold(tucker_to_tensor(tucker_tensor, skip_factor=skip_factor, transpose_factors=transpose_factors), mode)


def tucker_to_vec(tucker_tensor, skip_factor=None, transpose_factors=False):
    """Converts a Tucker decomposition into a vectorised tensor

    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices
    skip_factor : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a length of ``tensor.ndim``
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
    """
    return tensor_to_vec(tucker_to_tensor(tucker_tensor, skip_factor=skip_factor, transpose_factors=transpose_factors))


def tucker_mode_dot(tucker_tensor, matrix_or_vector, mode, keep_dim=False, copy=False):
        """n-mode product of a Tucker tensor and a matrix or vector at the specified mode

        Parameters
        ----------
        tucker_tensor : tl.TuckerTensor or (core, factors)
                        
        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        TuckerTensor = (core, factors)
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

        See also
        --------
        tucker_multi_mode_dot : chaining several mode_dot in one call
        """
        shape, rank = _validate_tucker_tensor(tucker_tensor)
        core, factors = tucker_tensor
        contract = False
        
        if tl.ndim(matrix_or_vector) == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                        shape, matrix_or_vector.shape, mode, shape[mode], matrix_or_vector.shape[1]
                    ))

        elif tl.ndim(matrix_or_vector) == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                        shape, matrix_or_vector.shape, mode, shape[mode], matrix_or_vector.shape[0]
                    ))
            if not keep_dim:
                contract = True # Contract over that mode
        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.')
                             
        if copy:
            factors = [tl.copy(f) for f in factors]
            core = tl.copy(core)
            #if not contract:
            #    core = tl.copy(core)
            #else:
            #    warnings.warn('copy=True and keepdim=False, while contracting with a vector'
            #                 ' will result in a new core with one less mode.')

        if contract:
            print('contracting mode')
            f = factors.pop(mode)
            core = mode_dot(core, tl.dot(matrix_or_vector, f), mode=mode)
        else:
            factors[mode] = tl.dot(matrix_or_vector, factors[mode])            

        return core, factors
        #return TuckerTensor(core, factors)
    