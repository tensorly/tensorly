from math import prod
from . import backend as tl


def tensor_to_vec(tensor):
    """Vectorises a tensor

    Parameters
    ----------
    tensor : ndarray
             tensor of shape ``(i_1, ..., i_n)``

    Returns
    -------
    1D-array
        vectorised tensor of shape ``(i_1 * i_2 * ... * i_n)``
    """
    return tl.reshape(tensor, (-1,))


def vec_to_tensor(vec, shape):
    """Folds a vectorised tensor back into a tensor of shape `shape`

    Parameters
    ----------
    vec : 1D-array
        vectorised tensor of shape ``(i_1 * i_2 * ... * i_n)``
    shape : tuple
        shape of the ful tensor

    Returns
    -------
    ndarray
        tensor of shape `shape` = ``(i_1, ..., i_n)``
    """
    return tl.reshape(vec, shape)


def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return tl.reshape(tl.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
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
    return tl.moveaxis(tl.reshape(unfolded_tensor, full_shape), 0, mode)


def partial_unfold(tensor, mode=0, skip_begin=1, skip_end=0, ravel_tensors=False):
    """Partially unfolds a tensor while ignoring the specified number of dimensions at the beginning and the end.

        For instance, if the first dimension of the tensor is the number of samples, to unfold each sample,
        set skip_begin=1.
        This would, for each i in ``range(tensor.shape[0])``, unfold ``tensor[i, ...]``.

    Parameters
    ----------
    tensor : ndarray
        tensor of shape n_samples x n_1 x n_2 x ... x n_i
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
        new_shape += [tensor.shape[-i] for i in range(1, 1 + skip_end)]

    return tl.reshape(tl.moveaxis(tensor, mode + skip_begin, skip_begin), new_shape)


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
    mode_dim = transposed_shape.pop(skip_begin + mode)
    transposed_shape.insert(skip_begin, mode_dim)
    return tl.moveaxis(
        tl.reshape(unfolded, transposed_shape), skip_begin, skip_begin + mode
    )


def partial_tensor_to_vec(tensor, skip_begin=1, skip_end=0):
    """Partially vectorises a tensor

        Partially vectorises a tensor while ignoring the specified dimension at the beginning and the end

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
    return partial_unfold(
        tensor, mode=0, skip_begin=skip_begin, skip_end=skip_end, ravel_tensors=True
    )


def partial_vec_to_tensor(matrix, shape, skip_begin=1, skip_end=0):
    """Refolds a partially vectorised tensor into a full one

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
    return partial_fold(
        matrix, mode=0, shape=shape, skip_begin=skip_begin, skip_end=skip_end
    )


def matricize(tensor, row_modes, column_modes=None):
    """Matricizes the given tensor

    Parameters
    ----------
    tensor : tl.tensor
    row_modes : tuple[int]
        modes to use as row of the matrix (in the desired order)
    column_modes : tuple[int], default is None
        modes to use as column of the matrix, in the desired order
        if None, the modes not in `row_modes` will be used in ascending order

    Returns
    -------
    matrix : tl.tensor of size (prod(tensor.shape[i] for i in row_modes), -1)
    """
    try:
        row_indices = list(row_modes)
    except TypeError:
        row_indices = [row_modes]

    if column_modes is None:
        column_indices = [i for i in range(tl.ndim(tensor)) if i not in row_indices]
    else:
        try:
            column_indices = list(column_modes)
        except TypeError:
            column_indices = [column_modes]
        if sorted(column_indices + row_indices) != list(range(tl.ndim(tensor))):
            msg = (
                "If you provide both column and row modes for the matricization"
                " then column_modes + row_modes must contain all the modes of the tensor."
                f" Yet, got row_modes={row_modes} and column_modes={column_modes}."
            )
            raise ValueError(msg)

    row_size = prod(tl.shape(tensor)[i] for i in row_indices)
    column_size = prod(tl.shape(tensor)[i] for i in column_indices)

    return tl.reshape(
        tl.transpose(tensor, row_indices + column_indices), (row_size, column_size)
    )
