"""
Core tensor operations with MXnet.
"""

# Author: Jean Kossaifi
# License: BSD 3 clause


# First check whether MXNet is installed
try:
    import mxnet as mx
except ImportError as error:
    message = ('Impossible to import MXNet.\n'
               'To use TensorLy with the MXNet backend, '
               'you must first install MXNet!')
    raise ImportError(message) from error

import warnings
import numpy
import scipy.linalg
import scipy.sparse.linalg

from numpy import testing
from mxnet import nd as nd
from mxnet.ndarray import arange, zeros, zeros_like, ones
from mxnet.ndarray import moveaxis, dot, transpose, reshape
from mxnet.ndarray import where, maximum, sign, prod

# Order 0 tensor, mxnet....
from math import sqrt as scalar_sqrt

from . import numpy_backend

def context(tensor):
    """Returns the context of a tensor
    """
    return {'ctx':tensor.context, 'dtype':tensor.dtype}

def tensor(data, ctx=mx.cpu(), dtype='float32'):
    """Tensor class
    """
    if dtype is None and isinstance(data, numpy.ndarray):
        dtype = data.dtype
    return nd.array(data, ctx=ctx, dtype=dtype)

def to_numpy(tensor):
    """Convert a tensor to numpy format

    Parameters
    ----------
    tensor : Tensor

    Returns
    -------
    ndarray
    """
    if isinstance(tensor, nd.NDArray):
        return tensor.asnumpy()
    elif isinstance(tensor, numpy.ndarray):
        return tensor
    else:
        return numpy.array(tensor)

def assert_array_equal(a, b, **kwargs):
    testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)

def assert_array_almost_equal(a, b, decimal=4, **kwargs):
    testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), decimal=decimal, **kwargs)

assert_raises = testing.assert_raises
assert_ = testing.assert_

def assert_equal(actual, desired, err_msg='', verbose=True):
    if isinstance(actual, nd.NDArray):
        actual =  actual.asnumpy()
        if actual.shape == (1, ):
            actual = actual[0]
    if isinstance(desired, nd.NDArray):
        desired =  desired.asnumpy()
        if desired.shape == (1, ):
            desired = desired[0]
    testing.assert_equal(actual, desired)



def shape(tensor):
    return tensor.shape

def ndim(tensor):
    return tensor.ndim

def kron(matrix1, matrix2):
    """Kronecker product"""
    s1, s2 = matrix1.shape
    s3, s4 = matrix2.shape
    return nd.reshape(
                matrix1.reshape((s1, 1, s2, 1))*matrix2.reshape((1, s3, 1, s4)),
                (s1*s3, s2*s4))

def solve(matrix1, matrix2):
    return tensor(numpy.linalg.solve(to_numpy(matrix1), to_numpy(matrix2)), **context(matrix1))

def min(tensor, *args, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.min(tensor, *args, **kwargs).asscalar()
    else:
        return numpy.min(tensor, *args, **kwargs)

def max(tensor, *args, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.max(tensor, *args, **kwargs).asscalar()
    else:
        return numpy.max(tensor, *args, **kwargs)

def abs(tensor, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.abs(tensor, **kwargs)
    else:
        return numpy.abs(tensor, **kwargs)

def norm(tensor, order=2, axis=None):
    """Computes the l-`order` norm of tensor

    Parameters
    ----------
    tensor : ndarray
    order : int
    axis : int or tuple

    Returns
    -------
    float or tensor
        If `axis` is provided returns a tensor.
    """
    # handle difference in default axis notation
    if axis is None:
        axis = ()

    if order == 'inf':
        res = nd.max(nd.abs(tensor), axis=axis)
    elif order == 1:
        res =  nd.sum(nd.abs(tensor), axis=axis)
    elif order == 2:
        res = nd.sqrt(nd.sum(tensor**2, axis=axis))
    else:
        res = nd.sum(nd.abs(tensor)**order, axis=axis)**(1/order)

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
    if len(matrices) < 2:
        raise ValueError('kr requires a list of at least 2 matrices, but {} given.'.format(len(matrices)))

    n_col = shape(matrices[0])[1]
    for i, e in enumerate(matrices[1:]):
        if not i:
            res = matrices[0]
        s1, s2 = shape(res)
        s3, s4 = shape(e)
        if not s2 == s4 == n_col:
            raise ValueError('All matrices should have the same number of columns.')
        res = reshape(reshape(res, (s1, 1, s2))*reshape(e, (1, s3, s4)),
                      (-1, n_col))
    return res

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
        contains the left singular vectors
    """
    # Check that matrix is... a matrix!
    if matrix.ndim != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
            matrix.ndim))

    # Choose what to do depending on the params
    ctx = context(matrix)
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
        return tensor(U, **ctx), tensor(S, **ctx), tensor(V, **ctx)

    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(numpy.dot(matrix, matrix.T.conj()), k=n_eigenvecs, which='LM')
            S = numpy.sqrt(S)
            V = numpy.dot(matrix.T.conj(), U * 1/S.reshape((1, -1)))
        else:
            S, V = scipy.sparse.linalg.eigsh(numpy.dot(matrix.T.conj(), matrix), k=n_eigenvecs, which='LM')
            S = numpy.sqrt(S)
            U = numpy.dot(matrix, V) * 1/S.reshape((1, -1))

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return tensor(U, **ctx), tensor(S, **ctx), tensor(V.T.conj(), **ctx)

def qr(matrix):
    try:
        # NOTE - should be replaced with geqrf when available
        Q, L = nd.linalg.gelqf(matrix.T)
        return Q.T, L.T
    except AttributeError:
        warnings.warn('This version of MXNet does not include the linear '
                      'algebra function gelqf(). Substituting with numpy.')
        Q, R = numpy_backend.qr(to_numpy(matrix))
        return tensor(Q), tensor(R)

def clip(tensor, a_min=None, a_max=None, indlace=False):
    if a_min is not None and a_max is not None:
        if indlace:
            nd.max(nd.min(tensor, a_max, out=tensor), a_min, out=tensor)
        else:
            tensor = nd.maximum(nd.minimum(tensor, a_max), a_min)
    elif min is not None:
        if indlace:
            nd.max(tensor, a_min, out=tensor)
        else:
            tensor = nd.maximum(tensor, a_min)
    elif max is not None:
        if indlace:
            nd.min(tensor, a_max, out=tensor)
        else:
            tensor = nd.minimum(tensor, a_max)
    return tensor

def all(tensor):
    return nd.sum(tensor != 0).asscalar()

def mean(tensor, axis=None, **kwargs):
    if axis is None:
        axis = ()
    res = nd.mean(tensor, axis=axis, **kwargs)
    if res.shape == (1,):
        return res.asscalar()
    else:
        return res

def sum(tensor, axis=None, **kwargs):
    if axis is None:
        axis = ()
    res = nd.sum(tensor, axis=axis, **kwargs)
    if res.shape == (1,):
        return res.asscalar()
    else:
        return res

def sqrt(tensor, *args, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.sqrt(tensor, *args, **kwargs)
    else:
        return scalar_sqrt(tensor)

def copy(tensor):
    return tensor.copy()

def concatenate(tensors, axis):
    return nd.concat(*tensors, dim=axis)
