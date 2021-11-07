"""Manipulation of matrices in the TT format"""

import tensorly as tl
from ._batched_tensordot import tensordot

def tt_matrix_to_tensor(tt_matrix):
    """Returns the full tensor whose TT-Matrix decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in TT-Matrix format
        into the corresponding full tensor

    Parameters
    ----------
    factors: list of 4D-arrays
              TT-Matrix factors (known as core) of shape (rank_k, left_dim_k, right_dim_k, rank_{k+1})

    Returns
    -------
    output_tensor: ndarray
                   tensor whose TT-Matrix decomposition was given by 'factors'
    """
    # Each core is of shape (rank_left, size_in, size_out, rank_right)
    _, in_shape, out_shape, _ = zip(*(tl.shape(f) for f in tt_matrix))
    ndim = len(in_shape)
    
    # Intertwine the dims 
    # full_shape = in_shape[0], out_shape[0], in_shape[1], ...
    full_shape = sum(zip(*(in_shape, out_shape)), ())
    order = list(range(0, ndim*2, 2)) + list(range(1, ndim*2, 2))

    for i, factor in enumerate(tt_matrix):
        if not i:
            res = factor
        else:
            res = tensordot(res, factor, ([-1], [0]))

    return tl.transpose(tl.reshape(res, full_shape), order)