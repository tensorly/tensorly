"""
Core tensor operations with PyTorch.
"""

import numpy
import scipy.linalg
import scipy.sparse.linalg
from numpy import testing
import torch
from . import numpy_backend

from torch import ones, zeros
from torch import max, min
from torch import sum, mean, abs, sqrt, sign, prod
from torch import matmul as dot
from torch import qr

# Order 0 tensor, mxnet....
from math import sqrt as scalar_sqrt

# Equivalent functions in pytorch 
maximum = max

#import numpy as np
# Author: Jean Kossaifi

# License: BSD 3 clause

def context(tensor):
    """Returns the context of a tensor
    """
    return {'dtype':tensor.type()}


def tensor(data, dtype=torch.FloatTensor):
    """Tensor class
    """
    if isinstance(data, numpy.ndarray):
        return torch.from_numpy(numpy.copy(data)).type(dtype)
    return torch.Tensor(data).type(dtype)

def to_numpy(tensor):
    """Convert a tensor to numpy format

    Parameters
    ----------
    tensor : Tensor

    Returns
    -------
    ndarray
    """
    if torch.is_tensor(tensor):
        return tensor.numpy()
    if isinstance(tensor, numpy.ndarray):
        return tensor

    try:
        return numpy.array(tensor)
    except ValueError:
        raise ValueError('Could not convert object of type {} into a Numpy '
                         'NDArray'.format(type(tensor)))


def assert_array_equal(a, b, **kwargs):
    testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)

def assert_array_almost_equal(a, b, decimal=3, **kwargs):
    testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), decimal=decimal, **kwargs)

assert_raises = testing.assert_raises
assert_equal = testing.assert_equal
assert_ = testing.assert_

def shape(tensor):
    return tensor.size()

def ndim(tensor):
    return tensor.dim()

def arange(start, stop=None, step=1.0):
    if stop is None:
        return torch.arange(start=0., end=float(start), step=float(step))
    else:
        return torch.arange(float(start), float(stop), float(step))

def reshape(tensor, shape):
    try:
        return tensor.view(*shape)
    except RuntimeError:
        return tensor.contiguous().view(*shape)

def clip(tensor, a_min=None, a_max=None, inplace=False):
    if a_max is None:
        a_max = torch.max(tensor)
    if a_min is None:
        a_min = torch.min(tensor)
    if inplace:
        return torch.clamp(tensor, a_min, a_max, out=tensor)
    else:
        return torch.clamp(tensor, a_min, a_max)

def where(condition, x, y):
    assert condition.shape == x.shape == y.shape, 'Dimension mismatch'
    result = zeros_like(condition, **context(x))
    result[condition] = x
    result[~condition] = y
    return result

def all(tensor):
    return torch.sum(tensor != 0)

def transpose(tensor):
    axes = list(range(ndim(tensor)))[::-1]
    return tensor.permute(*axes)

def zeros_like(tensor, dtype=torch.FloatTensor):
    return torch.zeros(tensor.size()).type(dtype)

def copy(tensor):
    return tensor.clone()

def moveaxis(tensor, source, target):
    axes = list(range(ndim(tensor)))
    try:
        axes.pop(source)
    except IndexError:
        raise ValueError('Source should verify 0 <= source < tensor.ndim'
                         'Got %d' % source)
    try:
        axes.insert(target, source)
    except IndexError:
        raise ValueError('Destination should verify 0 <= destination < tensor.ndim'
                         'Got %d' % target)
    return tensor.permute(*axes)

def kron(matrix1, matrix2):
    """Kronecker product"""
    s1, s2 = shape(matrix1)
    s3, s4 = shape(matrix2)
    return reshape(
        reshape(matrix1, (s1, 1, s2, 1))*reshape(matrix2, (1, s3, 1, s4)),
        (s1*s3, s2*s4))


def solve(matrix1, matrix2):
    solution, _ = torch.gesv(matrix2, matrix1)
    return solution


def norm(tensor, order=2, axis=None):
    """Computes the l-`order` norm of tensor.

    Parameters
    ----------
    tensor : ndarray
    order : int
    axis : int

    Returns
    -------
    float or tensor
        If `axis` is provided returns a tensor.
    """
    # pytorch does not accept `None` for any keyword arguments. additionally,
    # pytorch doesn't seems to support keyword arguments in the first place
    kwds = {}
    if axis is not None:
        kwds['dim'] = axis
    if order and order != 'inf':
        kwds['p'] = order

    if order == 'inf':
        res = torch.max(torch.abs(tensor), **kwds)
        if axis is not None:
            return res[0]  # ignore indices output
        return res
    return torch.norm(tensor, **kwds)

def mean(tensor, axis=None):
    if axis is None:
        return torch.mean(tensor)
    else:
        return torch.mean(tensor, dim=axis)

def sum(tensor, axis=None):
    if axis is None:
        return torch.sum(tensor)
    else:
        return torch.sum(tensor, dim=axis)

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
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    # Check that matrix is... a matrix!
    if ndim(matrix) != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
            ndim(matrix)))

    U, S, V = torch.svd(matrix, some=False)
    U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V.t()[:n_eigenvecs, :]
    return U, S, V

def sqrt(tensor, *args, **kwargs):
    if torch.is_tensor(tensor):
        return torch.sqrt(tensor, *args, **kwargs)
    else:
        return scalar_sqrt(tensor)

