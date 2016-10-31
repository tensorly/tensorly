import numpy as np
from scipy.linalg import svd

# Author: Jean Kossaifi

# License: BSD 3 clause



def soft_thresholding(tensor, threshold):
    """Soft-thresholding operator

        sign(tensor) * max[abs(tensor) - threshold, 0]

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
    >>> from tensorly.tenalg.proximal import soft_thresholding
    >>> tensor = np.array([[1, -2, 1.5], [-4, 3, -0.5]])
    >>> soft_thresholding(tensor, 1.1)
    array([[ 0. , -0.9,  0.4],
           [-2.9,  1.9,  0. ]])


    Example with missing values

    >>> mask = np.array([[0, 0, 1], [1, 0, 1]])
    >>> soft_thresholding(tensor, mask*1.1)
    array([[ 1. , -2. ,  0.4],
           [-2.9,  3. ,  0. ]])

    See also
    --------
    inplace_soft_thresholding : Inplace version of the soft-thresholding operator
    svd_thresholding : SVD-thresholding operator
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

    See also
    --------
    soft_thresholding : less memory-efficient but fast soft-thresholding operator
    svd_thresholding : SVD-thresholding operator
    """
    index_shrink = ((tensor <= threshold) & (tensor >= -threshold))
    index_more = (tensor > threshold)
    index_less = (tensor < -threshold)
    tensor[index_shrink] = 0
    tensor[index_more] -= threshold
    tensor[index_less] += threshold
    return tensor


def svd_thresholding(matrix, threshold):
    """Singular value thresholding operator

    Parameters
    ----------
    matrix : ndarray
    threshold : float

    Returns
    -------
    ndarray
        matrix on which the operator has been applied

    See also
    --------
    procrustes : procrustes operator
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


    See also
    --------
    svd_thresholding : SVD-thresholding operator
    """
    U, _, V = svd(matrix, full_matrices=False)
    return np.dot(U, V)

