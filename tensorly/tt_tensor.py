"""
Core operations on tensors in Tensor-Train (TT) format, also known as Matrix-Product-State (MPS)
"""

import tensorly as tl
from ._factorized_tensor import FactorizedTensor
from .utils import DefineDeprecated
import numpy as np
from scipy.optimize import brentq
import warnings

def _validate_tt_tensor(tt_tensor):
    factors = tt_tensor
    n_factors = len(factors)

    if isinstance(tt_tensor, TTTensor):
        # it's already been validated at creation
        return tt_tensor.shape, tt_tensor.rank
    elif isinstance(tt_tensor, (float, int)): #0-order tensor
        return 0, 0

    rank = []
    shape = []
    for index, factor in enumerate(factors):
        current_rank, current_shape, next_rank = tl.shape(factor)

        # Check that factors are third order tensors
        if not tl.ndim(factor)==3:
            raise ValueError('TT expresses a tensor as third order factors (tt-cores).\n'
                             'However, tl.ndim(factors[{}]) = {}'.format(
                                 index, tl.ndim(factor)))
        # Consecutive factors should have matching ranks
        if index and tl.shape(factors[index - 1])[2] != current_rank:
            raise ValueError('Consecutive factors should have matching ranks\n'
                             ' -- e.g. tl.shape(factors[0])[2]) == tl.shape(factors[1])[0])\n'
                             'However, tl.shape(factor[{}])[2] == {} but'
                             ' tl.shape(factor[{}])[0] == {} '.format(
                              index - 1, tl.shape(factors[index - 1])[2], index, current_rank))
        # Check for boundary conditions
        if (index == 0) and current_rank != 1:
            raise ValueError('Boundary conditions dictate factor[0].shape[0] == 1.'
                             'However, got factor[0].shape[0] = {}.'.format(
                              current_rank))
        if (index == n_factors - 1) and next_rank != 1:
            raise ValueError('Boundary conditions dictate factor[-1].shape[2] == 1.'
                             'However, got factor[{}].shape[2] = {}.'.format(
                              n_factors, next_rank))
    
        shape.append(current_shape)
        rank.append(current_rank)
        
    # Add last rank (boundary condition)
    rank.append(next_rank)
        
    return tuple(shape), tuple(rank)


def tt_to_tensor(factors):
    """Returns the full tensor whose TT decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in TT/Matrix-Product-State format
        into the corresponding full tensor

    Parameters
    ----------
    factors : list of 3D-arrays
              TT factors (TT-cores)

    Returns
    -------
    output_tensor : ndarray
                   tensor whose TT/MPS decomposition was given by 'factors'
    """
    if isinstance(factors, (float, int)): #0-order tensor
        return factors

    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return tl.reshape(full_tensor, full_shape)


def tt_to_unfolded(factors, mode):
    """Returns the unfolding matrix of a tensor given in TT (or Tensor-Train) format

    Reassembles a full tensor from 'factors' and returns its unfolding matrix
    with mode given by 'mode'

    Parameters
    ----------
    factors: list of 3D-arrays
              TT factors
    mode: int
          unfolding matrix to be computed along this mode

    Returns
    -------
    2-D array
    unfolding matrix at mode given by 'mode'
    """
    return tl.unfold(tt_to_tensor(factors), mode)


def tt_to_vec(factors):
    """Returns the tensor defined by its TT format ('factors') into
       its vectorized format

    Parameters
    ----------
    factors: list of 3D-arrays
              TT factors

    Returns
    -------
    1-D array
    vectorized format of tensor defined by 'factors'
    """
    return tl.tensor_to_vec(tt_to_tensor(factors))

def _tt_n_param(tensor_shape, rank):
    """Number of parameters of a MPS decomposition for a given `rank` and full `tensor_shape`.

    Parameters
    ----------
    tensor_shape : int tuple
        shape of the full tensor to decompose (or approximate)
    
    rank : tuple
        rank of the MPS decomposition
    
    Returns
    -------
    n_params : int
        Number of parameters of a MPS decomposition of rank `rank` of a full tensor of shape `tensor_shape`
    """
    factor_params = []
    for i, s in enumerate(tensor_shape):
        factor_params.append(rank[i]*s*rank[i+1])
    return np.sum(factor_params)


def validate_tt_rank(tensor_shape, rank='same', constant_rank=False, rounding='round',
                     allow_overparametrization=True):
    """Returns the rank of a TT Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, tuple, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple, just returns rank
    constant_rank : bool, default is False
        * if True, the *same* rank will be chosen for each modes
        * if False (default), the rank of each mode will be proportional to the corresponding tensor_shape

        *used only if rank == 'same' or 0 < rank <= 1*

    rounding = {'round', 'floor', 'ceil'}

    allow_overparametrization : bool, default is True
        if False, the rank must be realizable through iterative application of SVD 
        (used in tensorly.decomposition.tensor_train)

    Returns
    -------
    rank : int tuple
        rank of the decomposition
    """
    if rounding == 'ceil':
        rounding_fun = np.ceil
    elif rounding == 'floor':
        rounding_fun = np.floor
    elif rounding == 'round':
        rounding_fun = np.round
    else:
        raise ValueError(f'Rounding should be round, floor or ceil, but got {rounding}')

    if rank == 'same':
        rank = float(1)

    if isinstance(rank, float) and constant_rank:
        # Choose the *same* rank for each mode
        n_param_tensor = np.prod(tensor_shape)*rank
        order = len(tensor_shape)

        if order == 2:
            rank = (1, n_param_tensor / (tensor_shape[0] + tensor_shape[1]), 1)
            warnings.warn(f'Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor.')

        # R_k I_k R_{k+1} = R^2 I_k
        a = np.sum(tensor_shape[1:-1])
    
        # Border rank of 1, R_0 = R_N = 1
        # First and last factor of size I_0 R and I_N R
        b = np.sum(tensor_shape[0] + tensor_shape[-1])

        # We want the number of params of decomp (=sum of params of factors)
        # To be equal to c = \prod_k I_k
        c = -n_param_tensor
        delta = np.sqrt(b**2 - 4*a*c)

        # We get the non-negative solution
        solution = int(rounding_fun((- b + delta)/(2*a)))
        rank = rank=(1, ) + (solution, )*(order-1) + (1, )

    elif isinstance(rank, float):
        # Choose a rank proportional to the size of each mode
        # The method is similar to the above one for constant_rank == True
        order = len(tensor_shape)
        avg_dim = [(tensor_shape[i]+tensor_shape[i+1])/2 for i in range(order - 1)]
        if len(avg_dim) > 1:
            a = sum(avg_dim[i-1]*tensor_shape[i]*avg_dim[i] for i in range(1, order - 1))
        else:
            warnings.warn(f'Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor.')
            a = avg_dim[0]**2*tensor_shape[0]
        b = tensor_shape[0]*avg_dim[0] + tensor_shape[-1]*avg_dim[-1]
        c = -np.prod(tensor_shape)*rank
        delta = np.sqrt(b**2 - 4*a*c)

        # We get the non-negative solution
        fraction_param = (- b + delta)/(2*a)
        rank = tuple([max(int(rounding_fun(d*fraction_param)), 1) for d in avg_dim])
        rank = (1, ) + rank + (1, )

    else:
        # Check user input for potential errors
        n_dim = len(tensor_shape)
        if isinstance(rank, int):
            rank = [1] + [rank] * (n_dim-1) + [1]
        elif n_dim+1 != len(rank):
            message = 'Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {} while tl.ndim(tensor) + 1  = {}'.format(
                len(rank), n_dim + 1)
            raise(ValueError(message))

        # Initialization
        if rank[0] != 1:
            message = 'Provided rank[0] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[0] to 1.'.format(rank[0])
            raise ValueError(message)
        if rank[-1] != 1:
            message = 'Provided rank[-1] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[-1] to 1.'.format(rank[0])
            raise ValueError(message)
    
    if allow_overparametrization:
        return list(rank)
    else:
        validated_rank = [1]
        for i, s in enumerate(tensor_shape[:-1]):
            n_row = int(rank[i]*s)
            n_column = np.prod(tensor_shape[(i+1):]) # n_column of unfolding
            validated_rank.append(min(n_row, n_column, rank[i+1]))
        validated_rank.append(1)

        return validated_rank


class TTTensor(FactorizedTensor):
    def __init__(self, factors, inplace=False):
        super().__init__()
        
        # Will raise an error if invalid
        shape, rank = _validate_tt_tensor(factors)

        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors
    
    def __getitem__(self, index):
        return self.factors[index]
    
    def __setitem__(self, index, value):
        self.factors[index] = value
    
    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
        
    def __len__(self):
        return len(self.factors)
    
    def __repr__(self):
        message = 'factors list : rank-{} matrix-product-state tensor of shape {} '.format(
            self.rank, self.shape)
        return message
    
    def to_tensor(self):
        return tt_to_tensor(self)
    
    def to_unfolding(self, mode):
        return tt_to_unfolded(self, mode)
    
    def to_vec(self):
        return tt_to_vec(self)

mps_to_tensor = DefineDeprecated(deprecated_name='mps_to_tensor', use_instead=tt_to_tensor)
mps_to_unfolded = DefineDeprecated(deprecated_name='mps_to_unfolded', use_instead=tt_to_unfolded)
mps_to_vec = DefineDeprecated(deprecated_name='mps_to_vec', use_instead=tt_to_vec)
_validate_mps_tensor = DefineDeprecated(deprecated_name='_validate_mps_tensor', use_instead=_validate_tt_tensor)
