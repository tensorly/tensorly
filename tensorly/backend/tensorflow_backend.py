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

import numpy as np

from . import _generics
from .generic import kr, kron, partial_svd


backend = _generics.new_backend('tensorflow')

for name in ['float64', 'float32', 'int64', 'int32']:
    backend.register(getattr(np, name), name=name)

for name in ['ones', 'zeros', 'zeros_like', 'eye', 'reshape', 'transpose',
             'where', 'sign', 'abs', 'sqrt', 'qr']:
    backend.register(getattr(tf, name), name=name)

for name in ['min', 'max', 'mean', 'sum', 'prod', 'all']:
    backend.register(getattr(tf, 'reduce_' + name), name=name)

backend.register(tf.identity, 'copy')
backend.register(tf.concat, 'concatenate')

backend.register(kron)
backend.register(kr)
backend.register(partial_svd)


@backend.register
def context(tensor):
    return {'dtype': tensor.dtype}


@backend.register
def tensor(data, dtype=np.float32, device=None, device_id=None):
    if isinstance(data, tf.Tensor):
        return data

    if device is not None and device == 'GPU':
        return tf.constant(data, dtype=dtype).gpu(device_id)
    else:
        return tf.constant(data, dtype=dtype)


@backend.register
def is_tensor(tensor):
    isinstance(tensor, tf.Tensor)


@backend.register
def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    else:
        return tensor


@backend.register
def ndim(tensor):
    return len(tensor.get_shape()._dims)


@backend.register
def shape(tensor):
    return tuple(tensor.shape.as_list())


@backend.register
def arange(start, stop=None, step=1, dtype=np.float32):
    if stop is None:
        stop = start
        start = 0
    return tf.range(start=start, limit=stop, delta=step, dtype=dtype)


@backend.register
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


@backend.register
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


@backend.register
def norm(tensor, order=2, axis=None):
    if order == 'inf':
        order = np.inf
    res = tf.norm(tensor, ord=order, axis=axis)

    if res.shape == ():
        return res.numpy()
    return res


@backend.register
def dot(tensor1, tensor2):
    return tf.tensordot(tensor1, tensor2, axes=([ndim(tensor1) - 1], [0]))


@backend.register
def solve(lhs, rhs):
    squeeze = []
    if rhs.ndim == 1:
        squeeze = [-1]
        rhs = tf.reshape(rhs, (-1, 1))
    res = tf.matrix_solve(lhs, rhs)
    res = tf.squeeze(res, squeeze)
    return res


def truncated_svd(matrix, n_eigenvecs=None):
    """Computes an SVD on `matrix`

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

    S, U, V = tf.svd(matrix, full_matrices=full_matrices)
    U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], tf.transpose(V)[:n_eigenvecs, :]
    return U, S, V


SVD_FUNS = {'numpy_svd':partial_svd, 'truncated_svd':truncated_svd}
backend.register(SVD_FUNS, name='SVD_FUNS')


backend.register(np.testing.assert_raises)
backend.register(np.testing.assert_)


@backend.register
def assert_array_equal(a, b, **kwargs):
    np.testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)


@backend.register
def assert_array_almost_equal(a, b, decimal=3, **kwargs):
    np.testing.assert_array_almost_equal(to_numpy(a), to_numpy(b),
                                         decimal=decimal, **kwargs)


@backend.register
def assert_equal(actual, desired, err_msg='', verbose=True):
    if isinstance(actual, tf.Tensor):
        actual = actual.numpy()
        if actual.shape == (1, ):
            actual = actual[0]
    if isinstance(desired, tf.Tensor):
        desired = desired.numpy()
        if desired.shape == (1, ):
            desired = desired[0]
    np.testing.assert_equal(actual, desired)
