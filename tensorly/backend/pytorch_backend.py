"""
Core tensor operations with PyTorch.
"""

# Author: Jean Kossaifi
# License: BSD 3 clause


# First check whether PyTorch is installed
try:
    import torch
except ImportError as error:
    message = ('Impossible to import PyTorch.\n'
               'To use TensorLy with the PyTorch backend, '
               'you must first install PyTorch!')
    raise ImportError(message) from error

from distutils.version import LooseVersion

if LooseVersion(torch.__version__) < LooseVersion('0.4.0'):
    raise ImportError('You are using version="{}" of PyTorch.'
                      'Please update to "0.4.0" or higher.'.format(torch.__version__))

import numpy
import scipy.linalg
import scipy.sparse.linalg
from numpy import testing
from . import numpy_backend

from torch import ones, zeros, zeros_like, reshape, eye
from torch import max, min, where
from torch import sum, mean, abs, sqrt, sign, prod, sqrt
from torch import matmul as dot
from torch import qr

# Order 0 tensor, mxnet....
from math import sqrt as scalar_sqrt

# Equivalent functions in pytorch 
maximum = max

def context(tensor):
    """Returns the context of a tensor
    """
    return {'dtype':tensor.dtype, 'device':tensor.device, 'requires_grad':tensor.requires_grad}


def tensor(data, dtype=torch.float32, device='cpu', requires_grad=False):
    """Tensor class
    """

    # if the dtype is given as string or numpy.dtype class, we convert it to torch.dtype
    if dtype.__class__ == str or dtype.__class__ == numpy.dtype:
        if dtype == numpy.float16 or dtype == "float16":
            dtype = torch.half
        elif dtype == numpy.float32 or dtype == "float32":
            dtype = torch.float
        elif dtype == numpy.float64 or dtype == "float64" or dtype == numpy.float_ or dtype == "float_":
            dtype = torch.double
        elif dtype == numpy.uint8 or dtype == "uint8":
            dtype = torch.uint8
        elif dtype == numpy.int8 or dtype == "int8":
            dtype = torch.int8
        elif dtype == numpy.int16 or dtype == "int16":
            dtype = torch.short
        elif dtype == numpy.int32 or dtype == "int32":
            dtype = torch.int
        elif dtype == numpy.int64 or dtype == "int64" or dtype == numpy.int_ or dtype == "int_":
            dtype = torch.long
        else:
            raise ValueError('pytorch tensor dose not support dtype {} .'.format(dtype))

    if isinstance(data, numpy.ndarray):
        return torch.tensor(data.copy(), dtype=dtype, device=device, requires_grad=requires_grad)
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

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
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    elif isinstance(tensor, numpy.ndarray):
        return tensor

    try:
        return numpy.asarray(tensor)
    except ValueError:
        raise ValueError('Could not convert object of type {} into a Numpy '
                         'ndarray'.format(type(tensor)))

def assert_array_equal(a, b, **kwargs):
    testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)

def assert_array_almost_equal(a, b, decimal=3, **kwargs):
    testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), decimal=decimal, **kwargs)

assert_raises = testing.assert_raises

def assert_equal(actual, desired, err_msg='', verbose=True):
    if isinstance(actual, torch.Tensor):
        actual = to_numpy(actual)
    if isinstance(desired, torch.Tensor):
        desired = to_numpy(desired)
    testing.assert_equal(actual, desired, err_msg=err_msg, verbose=verbose)

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

def transpose(tensor, axes=None):
    if axes == None:
        axes = list(range(ndim(tensor)))[::-1]
    return tensor.permute(*axes)


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

def concatenate(tensors, axis=0):
    return torch.cat(tensors, dim=axis)

def argmin(input, axis=None):
        return torch.argmin(input, dim=axis)

def argmax(input, axis=None):
        return torch.argmax(input, dim=axis)


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

def _reverse(tensor, axis=0):
    """Reverses the elements along the specified dimension
    Parameters
    ----------
    tensor : tl.tensor
    axis : int, default is 0
        axis along which to reverse the ordering of the elements
    Returns
    -------
    reversed_tensor : for a 1-D tensor, returns the equivalent of
                      tensor[::-1] in NumPy
    """
    indices = torch.arange(tensor.shape[axis]-1, -1, -1, dtype=torch.int64)
    return tensor.index_select(axis, indices)

def truncated_svd(matrix, n_eigenvecs=None):
    """Computes a truncated SVD on `matrix` using pytorch's SVD
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
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if n_eigenvecs is None or n_eigenvecs > min_dim:
        full_matrices = True
    else:
        full_matrices = False

    U, S, V = torch.svd(matrix, some=full_matrices)
    U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
    return U, S, V

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
        raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(ndim(matrix)))
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
            message = ('trying to compute SVD with n_eigenvecs={}, which is larger than'
                       'max(matrix.shape)={1}. Setting n_eigenvecs to {1}'.format(
                           n_eigenvecs, max_dim))
            n_eigenvecs = max_dim
        # we compute decomposition on the largest of the two to keep more eigenvecs
        dim_1, dim_2 = dim_2, dim_1

    if dim_1 < dim_2:
        S, U = torch.symeig(dot(matrix, transpose(matrix)))
        S = sqrt(S)
        V = dot(transpose(matrix), U * 1/reshape(S, (1, -1)))
    else:
        S, V = torch.symeig(dot(transpose(matrix), matrix))
        S = sqrt(S)
        U = dot(matrix, V) * 1/reshape(S, (1, -1))

    U, S, V = _reverse(U, 1), _reverse(S), _reverse(transpose(V), 0)
    return U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]

def partial_svd(matrix, n_eigenvecs=None):
    """Computes a fast partial SVD on `matrix` using NumnPy
        if `n_eigenvecs` is specified, sparse eigendecomposition
        is used on either matrix.dot(matrix.T) or matrix.T.dot(matrix)
        Faster for very sparse svd (n_eigenvecs small) but uses numpy/scipy
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
    ctx = context(matrix)
    matrix = to_numpy(matrix)
    U, S, V = numpy_backend.partial_svd(matrix, n_eigenvecs)
    return tensor(U, **ctx), tensor(S, **ctx), tensor(V, **ctx)

SVD_FUNS = {'numpy_svd':partial_svd, 'truncated_svd':truncated_svd, 'symeig_svd':symeig_svd}
