"""
Core operations on tensors in Tensor-Train (TT) format, also known as Matrix-Product-State (MPS)
"""

import tensorly as tl
from ._factorized_tensor import FactorizedTensor
from .utils import DefineDeprecated
import numpy as np

def _validate_tt_tensor(tt_tensor):
    factors = tt_tensor
    n_factors = len(factors)
    
    if n_factors < 2:
        raise ValueError('A Tensor-Train (MPS) tensor should be composed of at least two factors and a core.'
                         'However, {} factor was given.'.format(n_factors))

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

        Re-assembles 'factors', which represent a tensor in TT/TT format
        into the corresponding full tensor

    Parameters
    ----------
    factors: list of 3D-arrays
              TT factors (known as core in TT terminology)

    Returns
    -------
    output_tensor: ndarray
                   tensor whose TT/TT decomposition was given by 'factors'
    """
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

def _validate_tt_rank(tensor_shape, rank='same', rounding='round'):
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
    rounding = {'round', 'floor', 'ceil'}

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

    if isinstance(rank, float) and (0 < rank <= 1):
        n_param_tensor = np.prod(tensor_shape)*rank
        order = len(tensor_shape)

        # Border rank of 1, R_0 = R_N = 1
        # First and last factor of size I_0 R and I_N R
        a = np.sum(tensor_shape[1:-1])
        # R_k I_k R_{k+1} = R^2 I_k
        b = np.sum(tensor_shape[0] + tensor_shape[-1])
        # We want the number of params of decomp (=sum of params of factors)
        # To be equal to c = \prod_k I_k
        c = -n_param_tensor
        delta = np.sqrt(b**2 - 4*a*c)
        # We get the non-negative solution
        solution = int(rounding_fun((- b + delta)/(2*a)))
        rank = rank=(1, ) + (solution, )*(order-1) + (1, )
    # else:
    #     raise ValueError(f'Got rank={rank}, expecting "same", a float or a rank.')
    return rank


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
