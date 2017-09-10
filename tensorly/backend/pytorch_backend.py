"""
Core tensor operations with PyTorch.
"""

import numpy
import scipy.linalg
import scipy.sparse.linalg
from numpy import testing
import torch
from . import numpy_backend

#import numpy as np
# Author: Jean Kossaifi

# License: BSD 3 clause

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
    elif isinstance(tensor, numpy.ndarray):
        return tensor
    else:
        raise ValueError('Only torch.Tensor or np.ndarray) are accepted,'
                         'given {}'.format(type(tensor)))


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
        return tensor.view(shape)
    except RuntimeError:
        return tensor.contiguous().view(shape)

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

def dot(matrix1, matrix2):
    output_shape = list(matrix1.shape)[:-1] + list(matrix2.shape)[1:]
    try:
        res = reshape(matrix1, (-1, matrix1.shape[-1])).mm(
                            reshape(matrix2, (matrix2.shape[0], -1)))
    except TypeError:
        matrix1 = matrix1.type(torch.FloatTensor) 
        matrix2 = matrix2.type(torch.FloatTensor) 
        res = reshape(matrix1, (-1, matrix1.shape[-1])).mm(
            reshape(matrix2, (matrix2.shape[0], -1)))

    return reshape(res, output_shape)

def kron(matrix1, matrix2):
    """Kronecker product"""
    return tensor(numpy.kron(to_numpy(matrix1), to_numpy(matrix2)))

def solve(matrix1, matrix2):
    return tensor(numpy.linalg.solve(to_numpy(matrix1), to_numpy(matrix2)))

def min(tensor, *args, **kwargs):
    return torch.min(tensor, *args, **kwargs)

def max(tensor, *args, **kwargs):
    return torch.min(tensor, *args, **kwargs)

def norm(tensor, order):
    """Computes the l-`order` norm of tensor
    Parameters
    ----------
    tensor : ndarray
    order : int
    Returns
    -------
    float
        l-`order` norm of tensor
    """
    if order == 'inf':
        return torch.max(torch.abs(tensor))
    if order == 1:
        res =  torch.sum(torch.abs(tensor))
    elif order == 2:
        res = numpy.sqrt(torch.sum(tensor**2))
    else:
        res = torch.sum(torch.abs(tensor)**order)**(1/order)
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
    if ndim(matrix) != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
            ndim(matrix)))

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
    if a_max is None:
        a_max = torch.max(tensor)
    if a_min is None:
        a_min = torch.min(tensor)
    if inplace:
        return torch.clamp(tensor, a_min, a_max, out=tensor)
    else:
        return torch.clamp(tensor, a_min, a_max)

def all(tensor):
    return torch.sum(tensor != 0)

def mean(tensor, *args, **kwargs):
    res = torch.mean(tensor, *args, **kwargs)
    return res

sqrt = torch.sqrt
abs = torch.abs
def sum(tensor, *args, **kwargs):
    res = torch.sum(tensor, *args, **kwargs)
    return res

def transpose(tensor):
    axes = list(range(ndim(tensor)))[::-1]
    return tensor.permute(*axes)

zeros = torch.zeros
def zeros_like(tensor):
    return torch.zeros(tensor.size())
ones = torch.ones
sign = torch.sign
maximum = torch.max
def copy(tensor):
    return tensor.clone()
