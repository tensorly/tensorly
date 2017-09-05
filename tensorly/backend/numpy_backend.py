"""
Core tensor operations.
"""

from numpy import testing
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

# Author: Jean Kossaifi

# License: BSD 3 clause

def tensor(data, dtype=np.float64):
    """Tensor class
    """
    return np.array(data, dtype=dtype)


def to_numpy(tensor):
    """Returns a copy of the tensor as a NumPy array"""
    return np.copy(tensor)

def assert_array_equal(a, b, **kwargs):
    return testing.assert_array_equal(a, b, **kwargs)

def assert_array_almost_equal(a, b, **kwargs):
    testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), **kwargs)

assert_raises = testing.assert_raises
assert_equal = testing.assert_equal
assert_ = testing.assert_

def shape(tensor):
    return tensor.shape

def ndim(tensor):
    return tensor.ndim

arange = np.arange
reshape = np.reshape
moveaxis = np.moveaxis
prod = np.prod
dot = np.dot
kron = np.kron
abs = np.abs
max = np.max
min = np.min
maximum = np.maximum
clip = np.clip
all = np.all
mean = np.mean
solve = np.linalg.solve
ones = np.ones
zeros = np.zeros
zeros_like = np.zeros_like
sum = np.sum
sign = np.sign
where = np.where
copy = np.copy

def norm(tensor, order=2, axis=None):
    """Computes the l-`order` norm of tensor
    Parameters
    ----------
    tensor : ndarray
    order : int
    axis : tuple
    Returns
    -------
    float
        l-`order` norm of tensor
    """
    # TODO: better handling of difference in null axis between numpy and mxnet
    if axis == ():
        axis = None

    if order == 'inf':
        return np.max(np.abs(tensor), axis=axis)
    if order == 1:
        return np.sum(np.abs(tensor), axis=axis)
    elif order == 2:
        return np.sqrt(np.sum(tensor**2, axis=axis))
    else:
        return np.sum(np.abs(tensor)**order, axis=axis)**(1/order)


def kr(matrices):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.

    Parameters
    ----------
    matrices : ndarray list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the product.

    Notes
    -----
    Mathematically:

    .. math::
         \\text{If every matrix } U_k \\text{ is of size } (I_k \\times R),\\\\
         \\text{Then } \\left(U_1 \\bigodot \\cdots \\bigodot U_n \\right) \\text{ is of size } (\\prod_{k=1}^n I_k \\times R)

    A more intuitive but slower implementation is::

        kr_product = np.zeros((n_rows, n_columns))
        for i in range(n_columns):
            cum_prod = matrices[0][:, i]  # Acuumulates the khatri-rao product of the i-th columns
            for matrix in matrices[1:]:
                cum_prod = np.einsum('i,j->ij', cum_prod, matrix[:, i]).ravel()
            # the i-th column corresponds to the kronecker product of all the i-th columns of all matrices:
            kr_product[:, i] = cum_prod

        return kr_product
    """
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


def partial_svd(matrix, n_eigenvecs=None):
    """Computes a fast partial SVD on `matrix`

        if `n_eigenvecs` is specified, sparse eigendecomposition
        is used on either matrix.dot(matrix.T) or matrix.T.dot(matrix)

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    U : 2D-array
        of shape (matrix.shape[0], n_eigenvecs)
        contains the right singular vectors
    S : 1D-array
        of shape (n_eigenvecs, )
        contains the singular values of `matrix`
    V : 2D-array
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    # Check that matrix is... a matrix!
    if matrix.ndim != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
            matrix.ndim))

    # Choose what to do depending on the params
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if n_eigenvecs is None or n_eigenvecs >= min_dim:
        # Default on standard SVD
        U, S, V = scipy.linalg.svd(matrix)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        return U, S, V

    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(np.dot(matrix, matrix.T), k=n_eigenvecs, which='LM')
            S = np.sqrt(S)
            V = np.dot(matrix.T, U * 1/S[None, :])
        else:
            S, V = scipy.sparse.linalg.eigsh(np.dot(matrix.T, matrix), k=n_eigenvecs, which='LM')
            S = np.sqrt(S)
            U = np.dot(matrix, V) * 1/S[None, :]

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return U, S, V.T

