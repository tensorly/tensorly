import numpy as np

# Author: Jean Kossaifi


def tensor_from_frontal_slices(*matrices):
    """Creates a third order tensor from a list of matrices (frontal slices)

    Parameters
    ----------
    matrices : ndarray list
        list of frontal slices, each a matrix of shape (I, J)

    Returns
    -------
    ndarray
        tensor of shape (I, J, len(matrix_list))
    """
    return np.concatenate([i[..., None] for i in matrices], axis=-1)


def tensor_to_vec(tensor):
    """Vectorises a tensor

    Parameters
    ----------
    tensor : ndarray
             tensor of shape (i_1, ..., i_n)

    Returns
    -------
    1D-array
        vectorised tensor of shape (i_1 * ... * i_n)
    """
    return np.ravel(tensor)


def vec_to_tensor(vec, shape):
    """Folds a vectorised tensor back into a tensor of shape `shape`

    Parameters
    ----------
    vec : 1D-array
        vectorised tensor of shape (i_1 * ... * i_n)
    shape : tuple
        shape of the ful tensor

    Returns
    -------
    ndarray
        tensor of shape `shape` = (i_1, ..., i_n)
    """
    return np.reshape(vec, shape)


def unfold(tensor, mode=0):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in range(0, tensor.ndim)

    Returns
    -------
    ndarray
        unfolded_tensor of shape (tensor.shape[mode], -1)
    """
    return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """Refolds the `mode`-mode unfolding into a tensor of shape `shape`

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape (shape[mode], -1)
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(unfolded_tensor.reshape(full_shape), 0, mode)


def partial_unfold(tensor, mode=0, skip_begin=1, skip_end=0, ravel_tensors=False):
    """Unfolds each tensor while ignoring the specified number of dimensions at the beginning and the end.

        For instance, if the first dimension of the tensor is the number of samples, to unfold each sample, you would
        set skip_begin=1. This would, for each `i in range(tensor.shape[0])`, unfold `tensor[i, ...]`.

    Parameters
    ----------
    tensor : ndarray
        tensor of shape n_samples*n_1*...*n_i
    mode : int
        indexing starts at 0, therefore mode is in range(0, tensor.ndim)
    skip_begin : int, optional
        number of dimensions to leave untouched at the beginning
    skip_end : int, optional
        number of dimensions to leave untouched at the end
    ravel_tensors : bool, optional
        if True, the unfolded tensors are also flattened

    Returns
    -------
    ndarray
        partially unfolded tensor
    """
    if ravel_tensors:
        new_shape = [-1]
    else:
        new_shape = [tensor.shape[mode + skip_begin], -1]

    if skip_begin:
        new_shape = [tensor.shape[i] for i in range(skip_begin)] + new_shape

    if skip_end:
        new_shape += [tensor.shape[-i] for i in range(skip_end)]

    return np.moveaxis(tensor, mode+skip_begin, skip_begin).reshape(new_shape)


def partial_fold(unfolded, mode, shape, skip_begin=1, skip_end=0):
    """Re-folds a partially unfolded tensor

    Parameters
    ----------
    unfolded : ndarray
        a partially unfolded tensor
    mode : int
        indexing starts at 0, therefore mode is in range(0, tensor.ndim)
    shape : tuple
        the shape of the original full tensor (including skipped dimensions)
    skip_begin : int, optional, default is 1
        number of dimensions to leave untouched at the beginning
    skip_end : int, optional
        number of dimensions to leave untouched at the end

    Returns
    -------
    ndarray
        partially re-folded tensor
    """
    transposed_shape = list(shape)
    mode_dim = transposed_shape.pop(skip_begin+mode)
    transposed_shape.insert(skip_begin, mode_dim)

    return np.moveaxis(unfolded.reshape(transposed_shape), skip_begin, skip_begin+mode)


def partial_tensor_to_vec(tensor, skip_begin=1, skip_end=0):
    """Partially vectorises a tensor

        Vectorises each tensor ignoring the specified dimension at the beginning and the end

    Parameters
    ----------
    tensor : ndarray
        tensor to partially vectorise
    skip_begin : int, optional, default is 1
        number of dimensions to leave untouched at the beginning
    skip_end : int, optional
        number of dimensions to leave untouched at the end

    Returns
    -------
    ndarray
        partially vectorised tensor with the `skip_begin` first and `skip_end` last dimensions untouched
    """
    return partial_unfold(tensor, mode=0, skip_begin=skip_begin, skip_end=skip_end, ravel_tensors=True)


def partial_vec_to_tensor(matrix, shape, skip_begin=1, skip_end=0):
    """Partially reconverts a partially vectorised tensor into a full one

    Parameters
    ----------
    matrix : ndarray
        a partially vectorised tensor
    shape : tuple
        the shape of the original full tensor (including skipped dimensions)
    skip_begin : int, optional, default is 1
        number of dimensions to leave untouched at the beginning
    skip_end : int, optional
        number of dimensions to leave untouched at the end

    Returns
    -------
    ndarray
        full tensor
    """
    return partial_fold(matrix, mode=0, shape=shape, skip_begin=skip_begin, skip_end=skip_end)
