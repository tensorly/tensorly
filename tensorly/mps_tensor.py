"""
Core operations on tensors in Matrix Product State (MPS) format, also known as Tensor-Train (TT)
"""

import tensorly as tl


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

