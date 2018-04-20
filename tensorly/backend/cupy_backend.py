"""
Core tensor operations with CuPy.
"""

# Author: Jean Kossaifi

# License: BSD 3 clause

# First check whether cupy is installed
try:
    import cupy as cp
except ImportError as error:
    message = ('Impossible to import cupy.\n'
               'To use TensorLy with the cupy backend, '
               'you must first install cupy!')
    raise ImportError(message) from error


import scipy.linalg
import scipy.sparse.linalg
from numpy import testing
import numpy
import warnings

from cupy import reshape, moveaxis, where, copy, transpose
from cupy import arange, ones, zeros, zeros_like
from cupy import dot, kron, concatenate
from cupy import max, min, maximum, all, mean, sum, sign, abs, prod, sqrt
from cupy.linalg import qr


def context(tensor):
    """Returns the context of a tensor
    """
    return {'dtype':tensor.dtype}

def tensor(data, dtype=numpy.float64):
    """Tensor class
    """
    return cp.array(data, dtype=dtype)


def to_numpy(tensor):
    """Returns a copy of the tensor as a NumPy array"""
    if isinstance(tensor, cp.ndarray):
        return cp.asnumpy(tensor)
    return tensor

def assert_array_equal(a, b, **kwargs):
    testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)

def assert_array_almost_equal(a, b, decimal=4, **kwargs):
    testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), decimal=decimal, **kwargs)

assert_raises = testing.assert_raises
assert_ = testing.assert_

def assert_equal(actual, desired, err_msg='', verbose=True):
    if isinstance(actual, cp.ndarray):
        actual =  to_numpy(actual)
        if actual.shape == (1, ):
            actual = actual[0]
    if isinstance(desired, cp.ndarray):
        desired =  to_numpy(desired)
        if desired.shape == (1, ):
            desired = desired[0]
    testing.assert_equal(actual, desired)


def shape(tensor):
    return tensor.shape

def ndim(tensor):
    return tensor.ndim

def clip(tensor, a_min=None, a_max=None, inplace=False):
    return cp.clip(tensor, a_min, a_max)

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
    if axis == ():
        axis = None

    if order == 'inf':
        res = cp.max(cp.abs(tensor), axis=axis)
    elif order == 1:
        res = cp.sum(cp.abs(tensor), axis=axis)
    elif order == 2:
        res = cp.sqrt(cp.sum(tensor**2, axis=axis))
    else:
        res = cp.sum(cp.abs(tensor)**order, axis=axis)**(1/order)

    if res.shape == ():
        return to_numpy(res)
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


def kron(matrix1, matrix2):
    """Kronecker product"""
    s1, s2 = matrix1.shape
    s3, s4 = matrix2.shape
    return cp.reshape(
                matrix1.reshape((s1, 1, s2, 1))*matrix2.reshape((1, s3, 1, s4)),
                (s1*s3, s2*s4))
     

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

def solve(matrix1, matrix2):
    try:
        cp.linalg.solve(matrix1, matrix2)
    except cp.cuda.cusolver.CUSOLVERError:
        warnings.warn('CuPy solver failed, using numpy.linalg.solve instead.')
        return tensor(numpy.linalg.solve(to_numpy(matrix1), to_numpy(matrix2)),
                      **context(matrix1))
