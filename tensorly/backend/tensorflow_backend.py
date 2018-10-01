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

from .core import Backend, register_backend


class TensorflowBackend(Backend):
    backend_name = 'tensorflow'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=np.float32, device=None, device_id=None):
        if isinstance(data, tf.Tensor):
            return data

        out = tf.constant(data, dtype=dtype)
        return out.gpu(device_id) if device == 'GPU' else out

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, tf.Tensor)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        else:
            return tensor

    @staticmethod
    def ndim(tensor):
        return len(tensor.get_shape()._dims)

    @staticmethod
    def shape(tensor):
        return tuple(tensor.shape.as_list())

    @staticmethod
    def arange(start, stop=None, step=1, dtype=np.float32):
        if stop is None:
            stop = start
            start = 0
        return tf.range(start=start, limit=stop, delta=step, dtype=dtype)

    def clip(self, tensor_, a_min=None, a_max=None, inplace=False):
        if a_min is not None:
            a_min = self.tensor(a_min, **self.context(tensor_))
        else:
            a_min = tf.reduce_min(tensor_)

        if a_max is not None:
            a_max = self.tensor(a_max, **self.context(tensor_))
        else:
            a_max = tf.reduce_max(tensor_)

        return tf.clip_by_value(tensor_, clip_value_min=a_min, clip_value_max=a_max)

    def moveaxis(self, tensor, source, target):
        axes = list(range(self.ndim(tensor)))
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

    @staticmethod
    def norm(tensor, order=2, axis=None):
        if order == 'inf':
            order = np.inf
        res = tf.norm(tensor, ord=order, axis=axis)

        if res.shape == ():
            return res.numpy()
        return res

    def dot(self, tensor1, tensor2):
        return tf.tensordot(tensor1, tensor2, axes=([self.ndim(tensor1) - 1], [0]))

    @staticmethod
    def solve(lhs, rhs):
        squeeze = []
        if rhs.ndim == 1:
            squeeze = [-1]
            rhs = tf.reshape(rhs, (-1, 1))
        res = tf.matrix_solve(lhs, rhs)
        res = tf.squeeze(res, squeeze)
        return res

    @staticmethod
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

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'truncated_svd': self.truncated_svd}


for name in ['float64', 'float32', 'int64', 'int32']:
    TensorflowBackend.register_method(name, getattr(np, name))

for name in ['ones', 'zeros', 'zeros_like', 'eye', 'reshape', 'transpose',
             'where', 'sign', 'abs', 'sqrt', 'qr']:
    TensorflowBackend.register_method(name, getattr(tf, name))

for name in ['min', 'max', 'mean', 'sum', 'prod', 'all']:
    TensorflowBackend.register_method(name, getattr(tf, 'reduce_' + name))

TensorflowBackend.register_method('copy', tf.identity)
TensorflowBackend.register_method('concatenate', tf.concat)


register_backend(TensorflowBackend())
