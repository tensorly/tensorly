"""
Core operations on tensors in Matrix Product State (MPS) format.
"""

import tensorly as tl


def mps_to_tensor(factors):
    """Returns the full tensor whose MPS decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in MPS format
        into the corresponding full tensor

    Parameters
    ----------
    factors: list of 3D-arrays
              MPS factors

    Returns
    -------
    output_tensor: ndarray
                   tensor whose MPS decomposition was given by 'factors'
    """

    n_mode_dimensions = mps_get_n_mode_dimensions(factors)

    D = len(n_mode_dimensions)

    (r0, n1, r1) = factors[0].shape
    output_tensor = factors[0]
    output_tensor = tl.reshape(output_tensor, (n1, r1))

    for k in range(1, D):

        (r_prev, n_k, r_k) = factors[k].shape
        G_k = tl.reshape(factors[k], (r_prev, n_k * r_k))

        output_tensor = tl.dot(output_tensor, G_k)
        output_tensor = tl.reshape(output_tensor, (-1, r_k))

    output_tensor = tl.reshape(output_tensor, n_mode_dimensions)

    return output_tensor


def mps_to_unfolded(factors, mode):
    """Returns the unfolding matrix of a tensor given in MPS format

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


def mps_get_mps_ranks(factors):
    """Returns the MPS ranks from 'factors'

    Parameters
    ----------
    factors: list of 3D-arrays
              MPS factors

    Returns
    -------
    List of ints
    MPS ranks of 'factors'
    """

    D = len(factors)
    mps_ranks = [None] * (D+1)

    for k in range(D):
        (r_prev, n_k, r_k) = factors[k].shape
        mps_ranks[k] = r_prev
        mps_ranks[k+1] = r_k

    return list(mps_ranks)


def mps_get_n_mode_dimensions(factors):
    """Returns the dimensions of the tensor defined by its MPS
       decomposition

    Parameters
    ----------
    factors: list of 3D-arrays
              MPS factors

    Returns
    -------
    List of ints
    Dimensions of each mode of the underlying tensor defined by 'factors'
    """

    D = len(factors)
    n_mode_dimensions = [None] * D

    for k in range(D):
        (_, n_k, _) = factors[k].shape
        n_mode_dimensions[k] = n_k

    return list(n_mode_dimensions)


def error_between_two_tensors(tensor1, tensor2):
    """Calculates the Frobenius norm of two tensors and returns the
       difference relative to the first tensor

    Parameters
    ----------
    tensor1: ndarray
    tensor2: ndarray

    Returns
    -------
    float
    Relative difference between 'tensor1' and 'tensor2' given by the Frobenius
    norm
    """


    return tl.norm(tensor1 - tensor2) / tl.norm(tensor1)
