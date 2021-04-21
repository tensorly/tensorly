from ... import backend as T
import numpy as np

# Author: Jean Kossaifi
# License: BSD 3 clause


def inner(tensor1, tensor2, n_modes=None):
    """Generalised inner products between tensors

        Takes the inner product between the last (respectively first)
        `n_modes` of `tensor1` (respectively `tensor2`)

    Parameters
    ----------
    tensor1, tensor2 : tensor
    n_modes : int, default is None
        * if None, the traditional inner product is returned (i.e. a float)
        * otherwise, the product between the `n_modes` last modes of `tensor1`
            and the `n_modes` first modes of `tensor2` is returned.
            The resulting tensor's order is `ndim(tensor1) - n_modes`.

    Returns
    -------
    inner_product : float if n_modes is None, tensor otherwise
    """
    # Traditional inner product
    if n_modes is None:
        if tensor1.shape != tensor2.shape:
            raise ValueError('Taking a generalised product between two tensors without specifying common modes'
                             ' is equivalent to taking inner product.'
                             'This requires tensor1.shape == tensor2.shape.'
                             'However, got tensor1.shape={} and tensor2.shape={}'.format(tensor1.shape, tensor2.shape))
        return T.sum(tensor1*tensor2)


    shape_t1 = list(T.shape(tensor1))
    shape_t2 = list(T.shape(tensor2))
    offset = len(shape_t1) - n_modes

    if shape_t1[offset:] != shape_t2[:n_modes]:
        raise ValueError('Incorrect shapes for inner product along {} common modes.'
                         'tensor_1.shape={}, tensor_2.shape={}'.format(n_modes, shape_t1, shape_t2))
    
    # Inner product along `n_modes` common modes
    start = ord('a')
    modes_t1 = ''.join(chr(i+start) for i in range(T.ndim(tensor1)))
    modes_t2 = ''.join(chr(i + start + offset) for i in range(T.ndim(tensor2)))
    output_modes = modes_t1[:offset] + modes_t2[n_modes:]

    equation = f'{modes_t1},{modes_t2}->{output_modes}'

    return T.einsum(equation, tensor1, tensor2)

