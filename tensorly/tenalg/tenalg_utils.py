from .. import backend as T
import warnings


def _validate_contraction_modes(shape1, shape2, modes, batched_modes=False):
    """Takes in the contraction modes (for a tensordot) and validates them

    Parameters
    ----------
    modes : int or tuple[int] or (modes1, modes2)
    batched_modes : bool, default is False

    Returns
    -------
    modes1, modes2 : a list of modes for each contraction
    """
    if isinstance(modes, int):
        if batched_modes:
            modes1 = [modes]
            modes2 = [modes]
        else:
            modes1 = list(range(-modes, 0))
            modes2 = list(range(0, modes))
    else:
        try:
            modes1, modes2 = modes
        except ValueError:
            modes1 = modes
            modes2 = modes
    try:
        modes1 = list(modes1)
    except TypeError:
        modes1 = [modes1]
    try:
        modes2 = list(modes2)
    except TypeError:
        modes2 = [modes2]

    if len(modes1) != len(modes2):
        if batched_modes:
            message = f"Both tensors must have the same number of batched modes"
        else:
            message = (
                "Both tensors must have the same number of modes to contract along. "
            )
        raise ValueError(
            message + f"However, got modes={modes}, "
            f" i.e. {len(modes1)} modes for tensor 1 and {len(modes2)} mode for tensor 2"
            f"(modes1={modes1}, and modes2={modes2})"
        )
    ndim1 = len(shape1)
    ndim2 = len(shape2)
    for i in range(len(modes1)):
        if shape1[modes1[i]] != shape2[modes2[i]]:
            if batched_modes:
                message = "Batch-dimensions must have the same dimensions in both tensors but got"
            else:
                message = "Contraction dimensions must have the same dimensions in both tensors but got"
            raise ValueError(
                message + f" mode {modes1[i]} of size {shape1[modes1[i]]} and "
                f" mode {modes2[i]} of size {shape2[modes2[i]]}."
            )
        if modes1[i] < 0:
            modes1[i] += ndim1
        if modes2[i] < 0:
            modes2[i] += ndim2

    return modes1, modes2


def _validate_khatri_rao(matrices, skip_matrix=None, reverse=False):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.
        (see [1]_ for more details).

        If one matrix only is given, that matrix is directly returned.

    Validate arguments. Common to both backends.
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    # Khatri-rao of only one matrix: just return that matrix
    if len(matrices) == 1:
        return matrices, matrices[0].shape[1]

    if T.ndim(matrices[0]) == 2:
        n_columns = matrices[0].shape[1]
    else:
        n_columns = 1
        matrices = [T.reshape(m, (-1, 1)) for m in matrices]
        warnings.warn(
            "Khatri-rao of a series of vectors instead of matrices. "
            "Condidering each has a matrix with 1 column."
        )

    # Optional part, testing whether the matrices have the proper size
    for i, matrix in enumerate(matrices):
        if T.ndim(matrix) != 2:
            raise ValueError(
                "All the matrices must have exactly 2 dimensions!"
                f"Matrix {i} has dimension {T.ndim(matrix)} != 2."
            )
        if matrix.shape[1] != n_columns:
            raise ValueError(
                "All matrices must have same number of columns!"
                f"Matrix {i} has {matrix.shape[1]} columns != {n_columns}."
            )

    if reverse:
        matrices = matrices[::-1]
        # Note: we do NOT use .reverse() which would reverse matrices even outside this function

    if len(matrices) < 2:
        raise ValueError(
            f"kr requires a list of at least 2 matrices, but {len(matrices)} given."
        )

    return matrices, n_columns
