from ... import backend as tl
from ...base import unfold, fold

from ... import unfold, fold, vec_to_tensor

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>

# License: BSD 3 clause


def mode_dot(tensor, matrix_or_vector, mode, transpose=False):
    """n-mode product of a tensor and a matrix or vector at the specified mode

    Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`


    Parameters
    ----------
    tensor : ndarray
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    matrix_or_vector : ndarray
        1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
        matrix or vectors to which to n-mode multiply the tensor
    mode : int

    Returns
    -------
    ndarray
        `mode`-mode product of `tensor` by `matrix_or_vector`
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
        * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

    See also
    --------
    multi_mode_dot : chaining several mode_dot in one call
    """
    # the mode along which to fold might decrease if we take product with a vector
    tensor_order = tl.ndim(tensor)
    start = ord('a')
    tensor_modes = ''.join(chr(start + i) for i in range(tensor_order))
    result_modes = [chr(start+tensor_order+1) if i == mode else j\
                            for i, j in enumerate(tensor_modes)]

    if tl.ndim(matrix_or_vector) == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1

        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[dim]
                ))
        if transpose: 
            matrix_or_vector = tl.conj(tl.transpose(matrix_or_vector))
        matrix_or_vector_modes = [chr(start+tensor_order+1), tensor_modes[mode]]

    elif tl.ndim(matrix_or_vector) == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                ))
        matrix_or_vector_modes = [tensor_modes[mode]]
        result_modes.pop(mode)

    else:
        raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                            'Provided array of dimension {} not in [1, 2].'.format(tl.ndim(matrix_or_vector)))

    result_modes = ''.join(result_modes)
    matrix_or_vector_modes = ''.join(matrix_or_vector_modes)
    equation = f'{tensor_modes},{matrix_or_vector_modes}->{result_modes}'
    return tl.einsum(equation, tensor, matrix_or_vector)


def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False):
    """n-mode product of a tensor and several matrices or vectors over several modes

    Parameters
    ----------
    tensor : ndarray

    matrix_or_vec_list : list of matrices or vectors of length ``tensor.ndim``

    skip : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a length of ``tensor.ndim``

    modes : None or int list, optional, default is None

    transpose : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    ndarray
        tensor times each matrix or vector in the list at mode `mode`

    Notes
    -----
    If no modes are specified, just assumes there is one matrix or vector per mode and returns:

    :math:`\\text{tensor  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`

    See also
    --------
    mode_dot
    """
    order = tl.ndim(tensor)

    if modes is None:
        modes = range(len(matrix_or_vec_list))

    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    # However, it is needed to pop dimensions contracted over
    factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])
    _, modes = zip(*factors_modes)

    start = ord('a')
    result_modes = [chr(start + i) for i in range(order)]
    tensor_modes = ''.join(result_modes)
    equation = ''
    counter = start + order + 1
    decrement = 0

    matrix_or_vec_list = []
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        #print(i, matrix_or_vec.shape, mode)
        if (skip is not None) and (i == skip):
            #print(f'skipping {skip}')
            continue

        if tl.ndim(matrix_or_vec) == 1:
            matrix_or_vec_list.append(matrix_or_vec)
            equation += f',{tensor_modes[mode]}'
            # We are contracting over the mode-th dimension
            result_modes.pop(mode - decrement)
            decrement += 1


        elif tl.ndim(matrix_or_vec) == 2:
            if transpose:
                matrix_or_vec_list.append(tl.conj(tl.transpose(matrix_or_vec)))
                # mat_symbol = f'{tensor_modes[mode]}{chr(counter)}'
            else:
                matrix_or_vec_list.append(matrix_or_vec)

            mat_symbol = f'{chr(counter)}{tensor_modes[mode]}'
            equation += ',' + mat_symbol
            # Contracting mode-th mode with a matrix: new dimension
            result_modes[mode - decrement] = chr(counter)
            counter += 1

        else:
            raise ValueError(f'Trying to contract a tensor with an {tl.ndim(matrix_or_vec)}--th'
                             f'order tensor along {mode}-th dimension.'
                             'Mode-dot only contracts a tensor with a vector or a matrix.')

    # If fully contracting 
    result_modes
    equation = tensor_modes + equation + f"->{''.join(result_modes)}"
    # matrix_or_vec_list = [m for (i, m) in enumerate(matrix_or_vec_list) if ((skip is None) or (skip != i))]

    #print(equation, tl.shape(tensor), [tl.shape(f) for f in matrix_or_vec_list])
    return tl.einsum(equation, tensor, *matrix_or_vec_list)
