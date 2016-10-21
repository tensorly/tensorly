import numpy as np
from scipy.linalg import svd

# Author: Jean Kossaifi


def soft_thresholding(tensor, threshold):
    """Soft thresholding operator

        sign(tensor)*max(abs(tensor) - threshold, 0)

    Parameters
    ----------
    tensor : ndarray
    threshold : float or ndarray with shape tensor.shape
        * If float the threshold is applied to the whole tensor
        * If ndarray, one theshold is applied per elements, 0 values are ignored

    Returns
    -------
    ndarray
        thresholded tensor on which the operator has been applied

    Examples
    --------
    Basic shrinkage

    >>> import numpy as np
    >>> from tensorlib.linalg import soft_thresholding
    >>> tensor = np.array([[1, -2, 1.5], [-4, 3, -0.5]])
    >>> soft_thresholding(tensor, 1.1)
    array([[ 0. , -0.9,  0.4],
           [-2.9,  1.9,  0. ]])


    Example with missing values

    >>> mask = np.array([[0, 0, 1], [1, 0, 1]])
    >>> soft_thresholding(tensor, mask*1.1)
    array([[ 1. , -2. ,  0.4],
           [-2.9,  3. ,  0. ]])

    """
    signs = np.sign(tensor)
    values = (signs*tensor - threshold)
    return np.where(values > 0, signs*values, 0)


def inplace_soft_thresholding(tensor, threshold):
    """Inplace version of the shrinkage operator

    Parameters
    ----------
    tensor : ndarray
    threshold : float

    Returns
    -------
    ndarray
        tensor on which the operator has been applied inplace

    Notes
    -----
    This version is memory efficient.
    For a faster but less memory efficient version, you can use this function:

    >>> def soft_thresholding(tensor, threshold):
    ...     return np.maximum(0, tensor - threshold) - np.maximum(0, -tensor - threshold)
    """
    index_shrink = ((tensor <= threshold) & (tensor >= -threshold))
    index_more = (tensor > threshold)
    index_less = (tensor < -threshold)
    tensor[index_shrink] = 0
    tensor[index_more] -= threshold
    tensor[index_less] += threshold
    return tensor


def svd_thresholing(matrix, threshold):
    """Singular value thresholding operator

    Parameters
    ----------
    matrix : ndarray
    threshold : float

    Returns
    -------
    ndarray
        matrix on which the operator has been applied
    """
    U, s, V = svd(matrix, full_matrices=False)
    return np.dot(U, soft_thresholding(s, threshold)[:, None]*V)


def procrustes(matrix):
    """Procrustes operator

    Parameters
    ----------
    matrix : ndarray

    Returns
    -------
    ndarray
        matrix on which the Procrustes operator has been applied
        has the same shape as the original tensor
    """
    U, _, V = svd(matrix, full_matrices=False)
    return np.dot(U, V)

