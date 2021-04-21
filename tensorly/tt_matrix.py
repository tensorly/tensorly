"""Module for matrices in the TT format"""

import tensorly as tl

# Note how tt_matrix_to_tensor is implemented in tenalg to allow for more efficient implementations
# (e.g. using the einsum backend)
from .tenalg import _tt_matrix_to_tensor as tt_matrix_to_tensor

from ._factorized_tensor import FactorizedTensor
import numpy as np

def validate_tt_matrix_rank(tensorized_shape, rank='same'):
    """Returns the rank of a TT-Matrix Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensorized matrix to decompose
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

    Returns
    -------
    rank : int tuple
        rank of the decomposition
    """

    n_dim = len(tensorized_shape) // 2

    if n_dim*2 != len(tensorized_shape):
        msg = (f'The order of the give tensorized shape is not a multiple of 2.'
                'However, there should be as many dimensions for the left side (number of rows)'
                ' as of the right side (number of columns). '
                ' For instance, to convert a matrix of size (8, 9) to the TT-format, '
                ' it can be tensorized to (2, 4, 3, 3) but NOT to (2, 2, 2, 3, 3).')
        raise ValueError(msg)

    left_shape = tensorized_shape[:n_dim]
    right_shape = tensorized_shape[n_dim:]

    full_shape = tuple(i*o for i, o in zip(left_shape, right_shape))
    return tl.tt_tensor.validate_tt_rank(full_shape, rank)

def _tt_matrix_n_param(tensorized_shape, rank):
    """Number of parameters of a TT-Matrix decomposition for a given `rank` and full `tensor_shape`.

    Parameters
    ----------
    tensorized_shape : int tuple
        shape of the full tensorized matrix to decompose (or approximate)
    
    rank : tuple
        rank of the TT-Matrix decomposition
    
    Returns
    -------
    n_params : int
        Number of parameters of a TT-Matrix decomposition of rank `rank` of a full tensor of shape `tensor_shape`
    """
    n_dim = len(tensorized_shape) // 2

    if n_dim*2 != len(tensorized_shape):
        msg = (f'The order of the give tensorized shape is not a multiple of 2.'
                'However, there should be as many dimensions for the left side (number of rows)'
                ' as of the right side (number of columns). '
                ' For instance, to convert a matrix of size (8, 9) to the TT-format, '
                ' it can be tensorized to (2, 4, 3, 3) but NOT to (2, 2, 2, 3, 3).')
        raise ValueError(msg)

    left_shape = tensorized_shape[:n_dim]
    right_shape = tensorized_shape[n_dim:]

    factor_params = []
    for i, (ls, rs) in enumerate(zip(left_shape, right_shape)):
        factor_params.append(rank[i]*ls*rs*rank[i+1])

    return np.sum(factor_params)

def tt_matrix_to_matrix(tt_matrix):
    """Reconstruct the original matrix that was tensorized and compressed in the TT-Matrix format

        Re-assembles 'factors', which represent a tensor in TT-Matrix format
        into the corresponding matrix

    Parameters
    ----------
    factors: list of 4D-arrays
              TT-Matrix factors (known as core) of shape (rank_k, left_dim_k, right_dim_k, rank_{k+1})

    Returns
    -------
    output_matrix: 2D-array
        matrix whose TT-Matrix decomposition was given by 'factors'
    """
    in_shape = tuple(c.shape[1] for c in tt_matrix)
    return tl.reshape(tt_matrix_to_tensor(tt_matrix), (np.prod(in_shape), -1))

def tt_matrix_to_unfolded(tt_matrix, mode):
    """Returns the unfolding matrix of a tensor given in TT-Matrix format

    Reassembles a full tensor from 'factors' and returns its unfolding matrix
    with mode given by 'mode'

    Parameters
    ----------
    factors : list of 3D-arrays
        TT-Matrix factors
    mode : int
        unfolding matrix to be computed along this mode

    Returns
    -------
    2-D array
    unfolding matrix at mode given by 'mode'
    """
    return tl.unfold(tt_matrix_to_tensor(tt_matrix), mode)

def tt_matrix_to_vec(tt_matrix):
    """Returns the tensor defined by its TT-Matrix format ('factors') into
       its vectorized format

    Parameters
    ----------
    factors : list of 3D-arrays
        TT factors

    Returns
    -------
    1-D array
        format of tensor defined by 'factors'
    """
    return tl.tensor_to_vec(tt_matrix_to_tensor(tt_matrix))

def _validate_tt_matrix(tt_tensor):
    factors = tt_tensor
    n_factors = len(factors)
    
    if n_factors < 1:
        raise ValueError('A Tensor-Train (MPS) tensor should be composed of at least one factor.'
                         'However, {} factor was given.'.format(n_factors))

    rank = []
    left_shape = []
    right_shape = []
    for index, factor in enumerate(factors):
        current_rank, current_left_shape, current_right_shape, next_rank = tl.shape(factor)

        # Check that factors are third order tensors
        if not tl.ndim(factor)==4:
            raise ValueError('A TTMatrix expresses a tensor as fourth order factors (tt-cores).\n'
                             'However, tl.ndim(factors[{}]) = {}'.format(
                                 index, tl.ndim(factor)))
        # Consecutive factors should have matching ranks
        if index and tl.shape(factors[index - 1])[-1] != current_rank:
            raise ValueError('Consecutive factors should have matching ranks\n'
                             ' -- e.g. tl.shape(factors[0])[-1]) == tl.shape(factors[1])[0])\n'
                             'However, tl.shape(factor[{}])[-1] == {} but'
                             ' tl.shape(factor[{}])[0] == {} '.format(
                              index - 1, tl.shape(factors[index - 1])[-1], index, current_rank))
        # Check for boundary conditions
        if (index == 0) and current_rank != 1:
            raise ValueError('Boundary conditions dictate factor[0].shape[0] == 1.'
                             'However, got factor[0].shape[0] = {}.'.format(
                              current_rank))
        if (index == n_factors - 1) and next_rank != 1:
            raise ValueError('Boundary conditions dictate factor[-1].shape[2] == 1.'
                             'However, got factor[{}].shape[2] = {}.'.format(
                              n_factors, next_rank))
    
        left_shape.append(current_left_shape)
        right_shape.append(current_right_shape)

        rank.append(current_rank)
        
    # Add last rank (boundary condition)
    rank.append(next_rank)
        
    return tuple(left_shape) + tuple(right_shape), tuple(rank)


class TTMatrix(FactorizedTensor):
    def __init__(self, factors, inplace=False):
        super().__init__()
        
        # Will raise an error if invalid
        shape, rank = _validate_tt_matrix(factors)

        self.shape = tuple(shape)
        self.order = len(self.shape) // 2
        self.left_shape = self.shape[:self.order]
        self.right_shape = self.shape[self.order:]
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
        message = (f'factors list : rank-{self.rank} TT-Matrix of tensorized shape {self.shape}'
                   f' corresponding to a matrix of size {np.prod(self.left_shape)} x {np.prod(self.right_shape)}')
        return message
    
    def to_tensor(self):
        return tt_matrix_to_tensor(self)
    
    def to_matrix(self):
        return tt_matrix_to_matrix(self)

    def to_unfolding(self, mode):
        return tt_matrix_to_unfolded(self, mode)
    
    def to_vec(self):
        return tt_matrix_to_vec(self)
