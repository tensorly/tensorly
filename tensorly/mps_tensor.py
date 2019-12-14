"""
Core operations on tensors in Matrix Product State (MPS) format, also known as Tensor-Train (TT)
"""

import tensorly as tl


def _validate_mps_tensor(mps_tensor):
    factors = mps_tensor
    n_factors = len(factors)
    
    if n_factors < 2:
        raise ValueError('A Matrix-Product-State (ttrain) tensor should be composed of at least two factors and a core.'
                         'However, {} factor was given.'.format(n_factors))

    rank = []
    shape = []
    for index, factor in enumerate(factors):
        current_rank, current_shape, next_rank = tl.shape(factor)

        # Check that factors are third order tensors
        if not tl.ndim(factor)==3:
            raise ValueError('MPS expresses a tensor as third order factors (tt-cores).\n'
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


def mps_to_tensor(factors):
    """Returns the full tensor whose MPS decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in MPS/TT format
        into the corresponding full tensor

    Parameters
    ----------
    factors: list of 3D-arrays
              MPS factors (known as core in TT terminology)

    Returns
    -------
    output_tensor: ndarray
                   tensor whose MPS/TT decomposition was given by 'factors'
    """
    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return tl.reshape(full_tensor, full_shape)


def mps_to_unfolded(factors, mode):
    """Returns the unfolding matrix of a tensor given in MPS (or Tensor-Train) format

    Reassembles a full tensor from 'factors' and returns its unfolding matrix
    with mode given by 'mode'

    Parameters
    ----------
    factors: list of 3D-arrays
              MPS factors
    mode: int
          unfolding matrix to be computed along this mode

    Returns
    -------
    2-D array
    unfolding matrix at mode given by 'mode'
    """
    return tl.unfold(mps_to_tensor(factors), mode)


def mps_to_vec(factors):
    """Returns the tensor defined by its MPS format ('factors') into
       its vectorized format

    Parameters
    ----------
    factors: list of 3D-arrays
              MPS factors

    Returns
    -------
    1-D array
    vectorized format of tensor defined by 'factors'
    """

    return tl.tensor_to_vec(mps_to_tensor(factors))

