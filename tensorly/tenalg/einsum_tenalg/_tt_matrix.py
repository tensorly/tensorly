"""Manipulation of matrices in the TT format"""

import tensorly as tl

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
    ndim = len(tt_matrix)
    order = list(range(0, ndim*2, 2)) + list(range(1, ndim*2, 2))

    start_in = ord('a')
    start_out = start_in + ndim
    start_rank = start_out + ndim
    factors_idx = []
    for i in range(ndim):
        idx = [start_rank+i, start_in+i, start_out+i, start_rank+i+1]
        factors_idx.append(''.join(chr(j) for j in idx))

    out_idx = ''.join(chr(start_in + i)+chr(start_out + i) for i in range(ndim)) 
    eq = ','.join(idx for idx in factors_idx) + '->' + out_idx
    
    res = tl.einsum(eq, *tt_matrix)
    
    return tl.tranpose(res, order)
    
