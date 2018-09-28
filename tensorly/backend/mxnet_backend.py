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

import math
import warnings

import numpy
from mxnet import nd
from mxnet.ndarray import reshape, dot, transpose

from . import _generics, numpy_backend
from .generic import kron, kr, partial_svd


backend = _generics.new_backend('mxnet')

for name in ['float64', 'float32', 'int64', 'int32']:
    backend.register(getattr(numpy, name), name=name)

for name in ['arange', 'zeros', 'zeros_like', 'ones', 'eye',
             'moveaxis', 'dot', 'transpose', 'reshape',
             'where', 'sign', 'prod']:
    backend.register(getattr(nd, name), name=name)

backend.register(kron)
backend.register(kr)
backend.register(partial_svd)


@backend.register
def context(tensor):
    return {'ctx': tensor.context, 'dtype': tensor.dtype}


@backend.register
def tensor(data, ctx=mx.cpu(), dtype=numpy.float32):
    if dtype is None and isinstance(data, numpy.ndarray):
        dtype = data.dtype
    return nd.array(data, ctx=ctx, dtype=dtype)


@backend.register
def is_tensor(tensor):
    return isinstance(tensor, nd.NDArray)


@backend.register
def to_numpy(tensor):
    if isinstance(tensor, nd.NDArray):
        return tensor.asnumpy()
    elif isinstance(tensor, numpy.ndarray):
        return tensor
    else:
        return numpy.array(tensor)


@backend.register
def shape(tensor):
    return tensor.shape


@backend.register
def ndim(tensor):
    return tensor.ndim


@backend.register
def solve(matrix1, matrix2):
    return tensor(numpy.linalg.solve(to_numpy(matrix1), to_numpy(matrix2)),
                  **context(matrix1))


@backend.register
def min(tensor, *args, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.min(tensor, *args, **kwargs).asscalar()
    else:
        return numpy.min(tensor, *args, **kwargs)


@backend.register
def max(tensor, *args, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.max(tensor, *args, **kwargs).asscalar()
    else:
        return numpy.max(tensor, *args, **kwargs)


@backend.register
def abs(tensor, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.abs(tensor, **kwargs)
    else:
        return numpy.abs(tensor, **kwargs)


@backend.register
def norm(tensor, order=2, axis=None):
    # handle difference in default axis notation
    if axis is None:
        axis = ()

    if order == 'inf':
        res = nd.max(nd.abs(tensor), axis=axis)
    elif order == 1:
        res = nd.sum(nd.abs(tensor), axis=axis)
    elif order == 2:
        res = nd.sqrt(nd.sum(tensor**2, axis=axis))
    else:
        res = nd.sum(nd.abs(tensor)**order, axis=axis)**(1 / order)

    if res.shape == (1,):
        return res.asscalar()

    return res


@backend.register
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


@backend.register
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


@backend.register
def all(tensor):
    return nd.sum(tensor != 0).asscalar()


@backend.register
def mean(tensor, axis=None, **kwargs):
    if axis is None:
        axis = ()
    res = nd.mean(tensor, axis=axis, **kwargs)
    if res.shape == (1,):
        return res.asscalar()
    else:
        return res


@backend.register
def sum(tensor, axis=None, **kwargs):
    if axis is None:
        axis = ()
    res = nd.sum(tensor, axis=axis, **kwargs)
    if res.shape == (1,):
        return res.asscalar()
    else:
        return res


@backend.register
def sqrt(tensor, *args, **kwargs):
    if isinstance(tensor, nd.NDArray):
        return nd.sqrt(tensor, *args, **kwargs)
    else:
        return math.sqrt(tensor)


@backend.register
def copy(tensor):
    return tensor.copy()


@backend.register
def concatenate(tensors, axis):
    return nd.concat(*tensors, dim=axis)


def symeig_svd(matrix, n_eigenvecs=None):
    """Computes a truncated SVD on `matrix` using symeig

        Uses symeig on matrix.T.dot(matrix) or its transpose

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
    if ndim(matrix) != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is %d != 2'
                         % ndim(matrix))

    dim_1, dim_2 = shape(matrix)
    if dim_1 <= dim_2:
        min_dim = dim_1
        max_dim = dim_2
    else:
        min_dim = dim_2
        max_dim = dim_1

    if n_eigenvecs is None:
        n_eigenvecs = max_dim

    if min_dim <= n_eigenvecs:
        if n_eigenvecs > max_dim:
            warnings.warn('Trying to compute SVD with n_eigenvecs={0}, which '
                          'is larger than max(matrix.shape)={1}. Setting '
                          'n_eigenvecs to {1}'.format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim
        # we compute decomposition on the largest of the two to keep more eigenvecs
        dim_1, dim_2 = dim_2, dim_1

    if dim_1 < dim_2:
        U, S = nd.linalg.syevd(dot(matrix, transpose(matrix)))
        S = sqrt(S)
        V = dot(transpose(matrix), U / reshape(S, (1, -1)))
    else:
        V, S = nd.linalg.syevd(dot(transpose(matrix), matrix))
        S = sqrt(S)
        U = dot(matrix, V) / reshape(S, (1, -1))

    U, S, V = U[:, ::-1], S[::-1], transpose(V)[::-1, :]
    return U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]


SVD_FUNS = {'numpy_svd': partial_svd,
            'symeig_svd': symeig_svd}
backend.register(SVD_FUNS, name='SVD_FUNS')
