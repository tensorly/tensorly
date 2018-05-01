"""
Core tensor operations with TensorFlow.
"""

# Author: Jean Kossaifi
# License: BSD 3 clause


# First check whether TensorFlow is installed
try:
    import tensorflow as tf
except ImportError as error:
    message = ('Impossible to import TensorFlow.\n'
               'To use TensorLy with the TensorFlow backend, '
               'you must first install TensorFlow!')
    raise ImportError(message) from error

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)

import numpy
import scipy.linalg
import scipy.sparse.linalg
from numpy import testing
from . import numpy_backend

from tensorflow import reshape, where, transpose
from tensorflow import ones, zeros, zeros_like
from tensorflow import sign, abs, sqrt
from tensorflow import qr
from tensorflow import concat as concatenate
from tensorflow import maximum
from tensorflow import minimum as min
from tensorflow import reduce_max as max
from tensorflow import reduce_mean as mean
from tensorflow import reduce_sum as sum
from tensorflow import reduce_prod as prod
from tensorflow import reduce_all as all
from tensorflow import reduce_all as all


def solve(lhs, rhs):
    squeeze = []
    if rhs.ndim == 1:
        squeeze = [-1]
        rhs = reshape(rhs, (-1, 1))
    res = tf.matrix_solve(lhs, rhs)
    res = tf.squeeze(res, squeeze)
    return res

def context(tensor):
    """Returns the context of a tensor
    """
    return {'dtype':tensor.dtype}

def tensor(data, dtype=numpy.float32, device=None, device_id=None):
    """Tensor class
    """
    if isinstance(data, tf.Tensor):
        return data
    else:
        if device is not None and device == 'GPU':
            return tf.constant(data, dtype=dtype).gpu(device_id)                                                                                                                                            
        else:
            return tf.constant(data, dtype=dtype)

def to_numpy(tensor):
    """Returns a copy of the tensor as a NumPy array"""
    if isinstance(tensor, numpy.ndarray):
        return tensor
    elif isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    else:
        return tensor

def assert_array_equal(a, b, **kwargs):
    testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)

def assert_array_almost_equal(a, b, decimal=3, **kwargs):
    testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), decimal=decimal, **kwargs)

assert_raises = testing.assert_raises
assert_ = testing.assert_

def assert_equal(actual, desired, err_msg='', verbose=True):
    if isinstance(actual, tf.Tensor):
        actual =  actual.numpy()
        if actual.shape == (1, ):
            actual = actual[0]
    if isinstance(desired, tf.Tensor):
        desired =  desired.numpy()
        if desired.shape == (1, ):
            desired = desired[0]
    testing.assert_equal(actual, desired)

def ndim(tensor):
    return len(tensor.get_shape()._dims)

def reshape(tensor, shape):
    return tf.reshape(tensor, shape)

def arange(start, stop=None, step=1, dtype=numpy.float32):
    if stop is None:
        stop = start
        start = 0
    return tf.range(start=start, limit=stop, delta=step, dtype=dtype)

def clip(tensor_, a_min=None, a_max=None, inplace=False):
    if a_min is not None:
        a_min = tensor(a_min, **context(tensor_))
    else:
        a_min = tf.reduce_min(tensor_)

    if a_max is not None:
        a_max = tensor(a_max, **context(tensor_))
    else:
        a_max = tf.reduce_max(tensor_)

    return tf.clip_by_value(tensor_, clip_value_min=a_min, clip_value_max=a_max)

def copy(tensor):
    return tf.identity(tensor)

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
    return tf.transpose(tensor, axes)

def kron(matrix1, matrix2):
    """Kronecker product"""
    s1, s2 = shape(matrix1)
    s3, s4 = shape(matrix2)
    return reshape(
        reshape(matrix1, (s1, 1, s2, 1))*reshape(matrix2, (1, s3, 1, s4)),
        (s1*s3, s2*s4))

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

    # Choose what to do depending on the params
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if n_eigenvecs is None or n_eigenvecs >= min_dim:
	# Default on standard SVD
        S, U, V = tf.svd(matrix, full_matrices=True)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], transpose(V)[:n_eigenvecs, :]
        return U, S, V
    
    else:
        ctx = context(matrix)
        matrix = to_numpy(matrix)
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
    if order == 'inf':                                                                                
        order = numpy.inf                                                                             
                                                                                                      
    res = tf.norm(tensor, ord=order, axis=axis)                                                       
                                                                                                      
    if res.shape == ():                                                                               
        return res.numpy()                                                                            
    return res

def dot(tensor1, tensor2):
    return tf.tensordot(tensor1, tensor2, axes=([ndim(tensor1) - 1], [0]))

def shape(tensor):
    # return tuple(s.value for s in tensor.shape)
    return tuple(tensor.shape.as_list())
