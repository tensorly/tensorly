"""
Core tensor operations with MXnet.
"""

import numpy
import scipy.linalg
import scipy.sparse.linalg
from numpy import testing
import mxnet as mx
from mxnet import nd as np
from . import numpy_backend

#import numpy as np
# Author: Jean Kossaifi

# License: BSD 3 clause

def tensor(data, ctx=mx.cpu(), dtype="float64"):
    """Tensor class
    """
    if dtype is None and isinstance(data, numpy.ndarray):
        dtype = data.dtype
    return np.array(data, ctx=ctx, dtype=dtype)

def to_numpy(tensor):
    """Convert a tensor to numpy format

    Parameters
    ----------
    tensor : Tensor

    Returns
    -------
    ndarray
    """
    if isinstance(tensor, np.NDArray):
        return tensor.asnumpy()
    elif isinstance(tensor, numpy.ndarray):
        return tensor
    else:
        raise ValueError('Only mx.nd.array or np.ndarray) are accepted,'
                         'given {}'.format(type(tensor)))


def assert_array_equal(a, b, **kwargs):
    testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)

def assert_array_almost_equal(a, b, decimal=4, **kwargs):
    testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), decimal=decimal, **kwargs)

assert_raises = testing.assert_raises
assert_equal = testing.assert_equal
assert_ = testing.assert_

def shape(tensor):
    return tensor.shape

def ndim(tensor):
    return tensor.ndim

def arange(start, stop=None, step=1.0):
    return np.arange(start, stop, step, dtype="float64")

def reshape(tensor, shape):
    return np.reshape(tensor, shape=shape)

def moveaxis(tensor, source, target):
    return np.moveaxis(tensor, source, target)

def prod(tensor, *args, **kwds):
    return np.prod(tensor, *args, **kwds)

def dot(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def kron(matrix1, matrix2):
    return tensor(numpy.kron(to_numpy(matrix1), to_numpy(matrix2)))

def solve(matrix1, matrix2):
    return tensor(numpy.linalg.solve(to_numpy(matrix1), to_numpy(matrix2)))

def min(tensor, *args, **kwargs):
    return np.min(tensor, *args, **kwargs).asscalar()

def max(tensor, *args, **kwargs):
    return np.min(tensor, *args, **kwargs).asscalar()

def norm(tensor, order=2, axis=None):
    """Computes the l-`order` norm of tensor
    Parameters
    ----------
    tensor : ndarray
    order : int
    axis : int or tuple
    Returns
    -------
    float
        l-`order` norm of tensor
    """
    # TODO: better handling of difference in null axis between numpy and mxnet
    if axis is None:
        axis = ()

    if order == 'inf':
        return np.max(np.abs(tensor), axis=axis).asscalar()
    if order == 1:
        res =  np.sum(np.abs(tensor), axis=axis)
    elif order == 2:
        res = np.sqrt(np.sum(tensor**2, axis=axis))
    else:
        res = np.sum(np.abs(tensor)**order, axis=axis)**(1./order)

    if res.shape == (1,):
        return res.asscalar()
    return res


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
    """
    return tensor(numpy_backend.kr([to_numpy(matrix) for matrix in matrices]))


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
    matrix = to_numpy(matrix)
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if n_eigenvecs is None or n_eigenvecs >= min_dim:
        # Default on standard SVD
        U, S, V = scipy.linalg.svd(matrix)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        return tensor(U), tensor(S), tensor(V)

    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(numpy.dot(matrix, matrix.T), k=n_eigenvecs, which='LM')
            S = numpy.sqrt(S)
            V = numpy.dot(matrix.T, U * 1/S.reshape((1, -1)))
        else:
            S, V = scipy.sparse.linalg.eigsh(numpy.dot(matrix.T, matrix), k=n_eigenvecs, which='LM')
            S = numpy.sqrt(S)
            U = numpy.dot(matrix, V) * 1/S.reshape((1, -1))

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return tensor(U), tensor(S), tensor(V.T)

def clip(tensor, a_min=None, a_max=None, inplace=False):
    if a_min is not None and a_max is not None:
        if inplace:
            tensor[:] = np.maximum(np.minimum(tensor, a_max), a_min)
        else:
            tensor = np.maximum(np.minimum(tensor, a_max), a_min)
    elif min is not None:
        if inplace:
            tensor[:] = np.maximum(tensor, a_min)
        else:
            tensor = np.maximum(tensor, a_min)
    elif min is not None:
        if inplace:
            tensor[:] = np.minimum(tensor, a_max)
        else:
            tensor = np.minimum(tensor, a_max)
    return tensor

def all(tensor):
    return np.sum(tensor != 0).asscalar()

def mean(tensor, *args, **kwargs):
    res = np.mean(tensor, *args, **kwargs)
    if res.shape == (1,):
        return res.asscalar()
    else:
        return res

sqrt = np.sqrt
abs = np.abs
def sum(tensor, *args, **kwargs):
    res = np.sum(tensor, *args, **kwargs)
    if res.shape == (1,):
        return res.asscalar()
    else:
        return res

zeros = np.zeros
zeros_like = np.zeros_like
ones = np.ones
sign = np.sign
where = np.where
maximum = np.maximum
def copy(tensor):
    return tensor.copy()
