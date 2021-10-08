"""
Core operations on Tucker tensors.
"""

from ._factorized_tensor import FactorizedTensor
from .base import unfold, tensor_to_vec
from .tenalg import multi_mode_dot, mode_dot
from . import backend as tl
import numpy as np
from scipy.optimize import brentq
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


def tucker_normalize(tucker_tensor):
    """Returns tucker_tensor with factors normalised to unit length with the normalizing constants absorbed into
    `core`.

    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices

    Returns
    -------
    TuckerTensor((core, factors))
    """
    core, factors = tucker_tensor
    normalized_factors = []
    for i, factor in enumerate(factors):
        scales = tl.norm(factor, axis=0)
        scales_non_zero = tl.where(scales == 0, tl.ones(tl.shape(scales), **tl.context(factor)), scales)
        core = core * tl.reshape(scales, (1,)*i + (-1, ) + (1,)*(tl.ndim(core) - i - 1))
        normalized_factors.append(factor / tl.reshape(scales_non_zero, (1, -1)))
    return TuckerTensor((core, normalized_factors))


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

        if contract:
            print('contracting mode')
            f = factors.pop(mode)
            core = mode_dot(core, tl.dot(matrix_or_vector, f), mode=mode)
        else:
            factors[mode] = tl.dot(matrix_or_vector, factors[mode])            

        return TuckerTensor((core, factors))


class TuckerTensor(FactorizedTensor):
    def __init__(self, tucker_tensor):
        super().__init__()

        shape, rank = _validate_tucker_tensor(tucker_tensor)
        core, factors = tucker_tensor
        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors
        self.core = core
    
    def __getitem__(self, index):
        if index == 0:
            return self.core
        elif index == 1:
            return self.factors
        else: 
            raise IndexError('You tried to access index {} of a Tucker tensor.\n'
                             'You can only access index 0 and 1 of a Tucker tensor'
                             '(corresponding respectively to core and factors)'.format(index))

    def __setitem__(self, index, value):
        if index == 0:
            self.core = value
        elif index == 1:
            self.factors = value
        else: 
            raise IndexError('You tried to set index {} of a Tucker tensor.\n'
                             'You can only set index 0 and 1 of a Tucker tensor'
                             '(corresponding respectively to core and factors)'.format(index))

    def __iter__(self):
        yield self.core
        yield self.factors
        
    def __len__(self):
        return 2
    
    def __repr__(self):
        message = 'Decomposed rank-{} TuckerTensor of shape {} '.format(self.rank, self.shape)
        return message
    
    def to_tensor(self):
        return tucker_to_tensor(self)
    
    def to_unfolded(self, mode):
        return tucker_to_unfolded(self, mode)
    
    def to_vec(self):
        return tucker_to_vec(self)

    def mode_dot(self, matrix_or_vector, mode, keep_dim=False, copy=False):
        """n-mode product with a matrix or vector at the specified mode

        Parameters
        ----------                        
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
        return tucker_mode_dot(self, matrix_or_vector, mode, keep_dim=keep_dim, copy=copy)


def _tucker_n_param(tensor_shape, rank):
    """Number of parameters of a Tucker decomposition for a given `rank` and full `tensor_shape`.

    Parameters
    ----------
    tensor_shape : int tuple
        shape of the full tensor to decompose (or approximate)
    
    rank : tuple
        rank of the Tucker decomposition
    
    Returns
    -------
    n_params : int
        Number of parameters of a Tucker decomposition of rank `rank` of a full tensor of shape `tensor_shape`
    """
    core_params = np.prod(rank)
    factors_params = np.sum([r*s for (r, s) in zip(rank, tensor_shape)])
    return core_params + factors_params


def validate_tucker_rank(tensor_shape, rank='same', rounding='round', fixed_modes=None):
    r"""Returns the rank of a Tucker Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, tuple, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple, just returns rank
    rounding = {'round', 'floor', 'ceil'}
    fixed_modes : int list or None, default is None
        if not None, a list of modes for which the rank will be the same as the original shape
        e.g. if i in fixed_modes, then rank[i] = tensor_shape[i]

    Returns
    -------
    rank : int tuple
        rank of the decomposition

    Notes
    -----
    For a fractional input rank, I want to find a Tucker rank such that: 
    n_param_decomposition = rank*n_param_tensor

    In particular, for an input of size I_1, ..., I_N:
    I find a value c such that the rank will be (c I_1, ..., c I_N)

    We have sn_param_tensor = I_1 x ... x I_N
    
    We look for a Tucker decomposition of rank (c I_1, ..., c I_N )
    This decomposition will have the following n_params:
    For the core : \prod_k c I_k = c^N \prod I_k = c^N n_param_tensor
    For the factors : \sum_k c I_k^2

    In other words we want to solve:
    c^N n_param_tensor + \sum_k c I_k^2 = rank*n_param_tensor
    """
    if rounding == 'ceil':
        rounding_fun = np.ceil
    elif rounding == 'floor':
        rounding_fun = np.floor
    elif rounding == 'round':
        rounding_fun = np.round
    else:
        raise ValueError(f'Rounding should be round, floor or ceil, but got {rounding}')

    # rank is 'same' or float: choose rank so as to preserve a fraction of the original #parameters
    if rank == 'same':
        rank = float(1)

    if isinstance(rank, float):
        n_modes_compressed = len(tensor_shape)
        n_param_tensor = np.prod(tensor_shape)
        
        if fixed_modes is not None:
            tensor_shape = list(tensor_shape)

            # sorted to be careful with the order when popping and reinserting to not remove/add at wrong index.
            # list (mode, shape) that we removed as they will be kept the same, rank[i] = 
            fixed_modes = [(mode, tensor_shape.pop(mode)) for mode in sorted(fixed_modes, reverse=True)][::-1]
            
            # number of parameters coming from the fixed modes (these don't have a variable size as a fun of fraction_param)
            n_fixed_params = np.sum([s**2 for _, s in fixed_modes]) # size of the factors
            n_modes_compressed -= len(fixed_modes)
        else:
            n_fixed_params = 0

        # Doesn't contain fixed_modes, those factors are accounted for in fixed_params
        squared_dims = np.sum([s**2 for s in tensor_shape])

        fun = lambda x : n_param_tensor*x**n_modes_compressed + squared_dims*x + n_fixed_params - rank*n_param_tensor
        fraction_param = brentq(fun, 0.0, max(rank, 1.0))
        rank = [max(int(rounding_fun(s*fraction_param)), 1) for s in tensor_shape]
        
        if fixed_modes is not None:
            for mode, size in fixed_modes:
                rank.insert(mode, size)
    
    elif isinstance(rank, int):
        n_modes = len(tensor_shape)
        message = "Given only one int for 'rank' for decomposition a tensor of order {}. Using this rank for all modes.".format(n_modes)
        warnings.warn(message, RuntimeWarning)
        if fixed_modes is None:
            rank = [rank]*n_modes
        else:
            rank = [rank  if i not in fixed_modes else s for (i, s) in enumerate(tensor_shape)]# *n_mode

    return rank

