"""
Core operations on tensors in Tensor Ring (TR) format
"""
import warnings

import numpy as np

import tensorly as tl
from ._factorized_tensor import FactorizedTensor


def tr_to_tensor(factors):
    """Returns the full tensor whose TR decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in TR format
        into the corresponding full tensor

    Parameters
    ----------
    factors : list of 3D-arrays
              TR factors (TR-cores)

    Returns
    -------
    output_tensor : ndarray
                   tensor whose TR decomposition was given by 'factors'
    """
    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (-1, factors[0].shape[2]))

    for factor in factors[1:-1]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    full_tensor = tl.reshape(full_tensor, (factors[-1].shape[2], -1, factors[-1].shape[0]))
    full_tensor = tl.moveaxis(full_tensor, 0, -1)
    full_tensor = tl.reshape(full_tensor, (-1, factors[-1].shape[0] * factors[-1].shape[2]))
    factor = tl.moveaxis(factors[-1], -1, 1)
    factor = tl.reshape(factor, (-1, full_shape[-1]))
    full_tensor = tl.dot(full_tensor, factor)
    return tl.reshape(full_tensor, full_shape)


def tr_to_unfolded(factors, mode):
    """Returns the unfolding matrix of a tensor given in TR format

    Reassembles a full tensor from 'factors' and returns its unfolding matrix
    with mode given by 'mode'

    Parameters
    ----------
    factors: list of 3D-arrays
              TR factors
    mode: int
          unfolding matrix to be computed along this mode

    Returns
    -------
    2-D array
    unfolding matrix at mode given by 'mode'
    """
    return tl.unfold(tr_to_tensor(factors), mode)


def tr_to_vec(factors):
    """Returns the tensor defined by its TR format ('factors') into
       its vectorized format

    Parameters
    ----------
    factors: list of 3D-arrays
              TR factors

    Returns
    -------
    1-D array
    vectorized format of tensor defined by 'factors'
    """
    return tl.tensor_to_vec(tr_to_tensor(factors))


def _validate_tr_tensor(tr_tensor):
    factors = tr_tensor
    n_factors = len(factors)

    if n_factors < 2:
        raise ValueError('A Tensor Ring tensor should be composed of at least two factors.'
                         'However, {} factor was given.'.format(n_factors))

    rank = []
    shape = []
    for index, factor in enumerate(factors):
        current_rank, current_shape, next_rank = tl.shape(factor)

        # Check that factors are third order tensors
        if not tl.ndim(factor) == 3:
            raise ValueError('TR expresses a tensor as third order factors (tr-cores).\n'
                             'However, tl.ndim(factors[{}]) = {}'.format(index, tl.ndim(factor)))

        # Consecutive factors should have matching ranks
        if tl.shape(factors[index - 1])[2] != current_rank:
            raise ValueError('Consecutive factors should have matching ranks\n'
                             ' -- e.g. tl.shape(factors[0])[2]) == tl.shape(factors[1])[0])\n'
                             'However, tl.shape(factor[{}])[2] == {} but'
                             ' tl.shape(factor[{}])[0] == {}'.format(
                index - 1, tl.shape(factors[index - 1])[2], index, current_rank))

        shape.append(current_shape)
        rank.append(current_rank)

    # Add last rank (boundary condition)
    rank.append(next_rank)

    return tuple(shape), tuple(rank)


def _tr_n_param(tensor_shape, rank):
    """Number of parameters of a TR decomposition for a given `rank` and full `tensor_shape`.

    Parameters
    ----------
    tensor_shape : int tuple
        shape of the full tensor to decompose (or approximate)

    rank : tuple
        rank of the TR decomposition

    Returns
    -------
    n_params : int
        Number of parameters of a TR decomposition of rank `rank` of a full tensor of shape `tensor_shape`
    """
    factor_params = []
    for i, s in enumerate(tensor_shape):
        factor_params.append(rank[i] * s * rank[i + 1])
    return np.sum(factor_params)


def validate_tr_rank(tensor_shape, rank='same', rounding='round'):
    """Returns the rank of a Tensor Ring Decomposition

    Parameters
    ----------
    tensor_shape : tuple
        shape of the tensor to decompose
    rank : {'same', float, tuple, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple, just returns rank
    rounding : {'round', 'floor', 'ceil'}

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

    n_dim = len(tensor_shape)
    if n_dim == 2:
        warnings.warn('Determining the TR-rank for the trivial case of a matrix'
                      f' (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor.')

    if isinstance(rank, float):
        # Choose the *same* rank for each mode
        n_param_tensor = np.prod(tensor_shape) * rank

        # R_k I_k R_{k+1} = R^2 I_k
        solution = int(rounding_fun(np.sqrt(n_param_tensor / np.sum(tensor_shape))))
        rank = (solution,) * (n_dim + 1)

    else:
        # Check user input for potential errors
        n_dim = len(tensor_shape)
        if isinstance(rank, int):
            rank = (rank,) * (n_dim + 1)
        elif n_dim + 1 != len(rank):
            message = ('Provided incorrect number of ranks. '
                       'Should verify len(rank) == tl.ndim(tensor)+1, '
                       f'but len(rank) = {len(rank)} while tl.ndim(tensor)+1 = {n_dim + 1}')
            raise ValueError(message)

        # Check first and last rank
        if rank[0] != rank[-1]:
            message = (f'Provided rank[0] == {rank[0]} and rank[-1] == {rank[-1]}'
                       ' but boundaring conditions dictatate rank[0] == rank[-1]')
            raise ValueError(message)

    return list(rank)


class TRTensor(FactorizedTensor):
    def __init__(self, factors):
        super().__init__()

        # Will raise an error if invalid
        shape, rank = _validate_tr_tensor(factors)

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
        message = 'factors list : rank-{} tensor ring tensor of shape {}'.format(
            self.rank, self.shape)
        return message

    def to_tensor(self):
        return tr_to_tensor(self)

    def to_unfolding(self, mode):
        return tr_to_unfolded(self, mode)

    def to_vec(self):
        return tr_to_vec(self)
