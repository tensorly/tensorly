import numpy as np
from ..base import fold, unfold

# Author: Jean Kossaifi


def mode_dot(tensor, matrix_or_vector, mode):
        """n-mode product of a tensor by a matrix at the specified mode.

        Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`


        Parameters
        ----------
        tensor : ndarray
            tensor of shape `(i_1, ..., i_k, ..., i_N)`
        matrix_or_vector : ndarray
            1D or 2D array of shape `(J, i_k)` or `(i_k, )`
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        ndarray
            - of shape `(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
            - of shape `(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector
        """
        new_shape = list(tensor.shape)

        if matrix_or_vector.ndim == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[1]
                    ))
            new_shape[mode] = matrix_or_vector.shape[0]
            factor_is_vec = False

        elif matrix_or_vector.ndim == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                    ))
            if len(new_shape) > 1:
                new_shape[mode] = 1
            else:
                new_shape = [1]
            factor_is_vec = True
        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                             'Provided array of dimension {} not in [1, 2].'.format(matrix_or_vector.ndim))

        res = np.dot(matrix_or_vector, unfold(tensor, mode))

        if factor_is_vec:
            return np.squeeze(fold(res, mode, new_shape))
        else:
            return fold(res, mode, new_shape)


def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False):
    """n-mode product of a tensor and several matrices or vectors

    Parameters
    ----------
    tensor : ndarray

    matrix_or_vec_list : list of matrices or vectors of lengh ``tensor.ndim``

    skip : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of `tensor.ndim`

    modes : None or int list, optional, default is None

    transpose : bool, optional, default is False
        if True, the matrices or vectors in in the list are transposed

    Returns
    -------
    ndarray
        tensor times each matrix or vector in the list

    Notes
    -----
    If no modes are specified, just assumes there is one matrix or vector per mode and returns:

    :math:`\\text{tensor  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`
    """
    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = tensor

    for i, (matrix_or_vec, mode) in enumerate(zip(matrix_or_vec_list, modes)):
        if (skip is not None) and (i == skip):
            continue

        if transpose:
            res = mode_dot(res, matrix_or_vec.T, mode - decrement)
        else:
            res = mode_dot(res, matrix_or_vec, mode - decrement)

        if matrix_or_vec.ndim == 1:
            decrement = 1

    return res