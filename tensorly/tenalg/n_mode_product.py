from .. import backend as T
from ..base import unfold, fold

from tensorly import unfold, fold, vec_to_tensor

def mode_dot(tensor, matrix_or_vector, mode):
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
        fold_mode = mode
        new_shape = list(tensor.shape)

        if T.ndim(matrix_or_vector) == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[1]
                    ))
            new_shape[mode] = matrix_or_vector.shape[0]
            vec = False

        elif T.ndim(matrix_or_vector) == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != tensor.shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                        tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                    ))
            if len(new_shape) > 1:
                new_shape.pop(mode)
            else:
                new_shape = [1]
            vec = True

        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                             'Provided array of dimension {} not in [1, 2].'.format(T.ndim(matrix_or_vector)))

        res = T.dot(matrix_or_vector, unfold(tensor, mode))

        if vec: # We contracted with a vector, leading to a vector
            return vec_to_tensor(res, shape=new_shape)
        else: # tensor times vec: refold the unfolding
            return fold(res, fold_mode, new_shape)


def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False):
    """n-mode product of a tensor and several matrices or vectors over several modes

    Parameters
    ----------
    tensor : ndarray

    matrix_or_vec_list : list of matrices or vectors of lengh ``tensor.ndim``

    skip : None or int, optional, default is None
        if not None, index of a matrix to skip
        Note that in any case, `modes`, if provided, should have a lengh of ``tensor.ndim``

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
    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = tensor

    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        if (skip is not None) and (i == skip):
            continue

        if transpose:
            res = mode_dot(res, T.transpose(matrix_or_vec), mode - decrement)
        else:
            res = mode_dot(res, matrix_or_vec, mode - decrement)

        if T.ndim(matrix_or_vec) == 1:
            decrement += 1

    return res

def unfolding_dot_khatri_rao(tensor, factors, mode):
    """mode-n unfolding times khatri-rao product of factors
    
    Parameters
    ----------
    tensor : tl.tensor
        tensor to unfold
    factors : tl.tensor list
        list of matrices of which to the khatri-rao product
    mode : int
        mode on which to unfold `tensor`
    
    Returns
    -------
    mttkrp
        dot(unfold(tensor, mode), khatri-rao(factors))

    Notes
    -----
    This is a variant of::
    
        unfolded = unfold(tensor, mode)
        kr_factors = khatri_rao(factors, skip_matrix=mode)
        mttkrp2 = tl.dot(unfolded, kr_factors)

    Multiplying with the Khatri-Rao product is equivalent to multiplying,
    for each rank, with the kronecker product of each factor. 
    In code::

        mttkrp_parts = []
        for r in range(rank):
            component = tl.tenalg.multi_mode_dot(tensor, [f[:, r] for f in factors], skip=mode)
            mttkrp_parts.append(component)
        mttkrp = tl.stack(mttkrp_parts, axis=1)
        return mttkrp 
    
    The same idea could be expressed using einsum::
    
        ndims = tl.ndim(tensor)
        tensor_idx = ''.join(chr(ord('a') + i) for i in range(ndims))
        rank = chr(ord('a') + ndims + 1)
        op = tensor_idx
        for i in range(ndims):
            if i != mode:
                op += ',' + ''.join([tensor_idx[i], rank])
            else:
                result = ''.join([tensor_idx[i], rank])
        op += '->' + result
        factors = [f for (i, f) in enumerate(factors) if i != mode]
        return tl_einsum(op, tensor, *factors)
    """
    projected = multi_mode_dot(tensor, factors, skip=mode, transpose=True)
    ndims = T.ndim(tensor)
    res = []
    for i in range(factors[0].shape[1]):
        index = tuple([slice(None) if k == mode  else i for k in range(ndims)])
        res.append(projected[index])
    return T.stack(res, axis=-1)
