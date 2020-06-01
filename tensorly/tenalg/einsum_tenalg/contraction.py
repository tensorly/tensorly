import numpy as np
from ... import backend as tl

# Author: Jean Kossaifi

# License: BSD 3 clause

def contract(tensor1, modes1, tensor2, modes2):
    """Tensor contraction between two tensors on specified modes
    
    Parameters
    ----------
    tensor1 : tl.tensor
    modes1 : int list or int
        modes on which to contract tensor1
    tensor2 : tl.tensor
    modes2 : int list or int
        modes on which to contract tensor2

    Returns
    -------
    contraction : tensor1 contracted with tensor2 on the specified modes
    """
    if isinstance(modes1, int):
        modes1 = [modes1]
    if isinstance(modes2, int):
        modes2 = [modes2]
    modes1 = list(modes1)
    modes2 = list(modes2)
    
    if len(modes1) != len(modes2):
        raise ValueError('Can only contract two tensors along the same number of modes'
                         '(len(modes1) == len(modes2))'
                         'However, got {} modes for tensor 1 and {} mode for tensor 2'
                         '(modes1={}, and modes2={})'.format(
                           len(modes1), len(modes2), modes1, modes2))
    
    contraction_dims = [tl.shape(tensor1)[i] for i in modes1]
    if contraction_dims != [tl.shape(tensor2)[i] for i in modes2]:
        raise ValueError('Trying to contract tensors over modes of different sizes'
                         '(contracting modes of sizes {} and {}'.format(
                             contraction_dims, [tl.shape(tensor2)[i] for i in modes2]))

    start = ord('a')
    order_t1 = tl.ndim(tensor1)
    all_modes1 = [chr(start+i) for i in range(order_t1)]
    all_modes2 = [chr(start+i+order_t1) for i in range(tl.ndim(tensor2))]

    for m1, m2 in zip(modes1, modes2):
        all_modes2[m2] = all_modes1[m1]

    remaining_modes1 = [j for i, j in enumerate(all_modes1) if i not in modes1]
    remaining_modes2 = [j for i, j in enumerate(all_modes2) if i not in modes2]
    remaining_modes = remaining_modes1 + remaining_modes2
    tostr = lambda x : ''.join(x)
    equation = f'{tostr(all_modes1)},{tostr(all_modes2)}->{tostr(remaining_modes)}'
    
    return tl.einsum(equation, tensor1, tensor2)
