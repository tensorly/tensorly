try:
    import tensorflow as tf
except ImportError as error:
    message = ('Impossible to import TensorFlow.\n'
               'To use TensorLy with the TensorFlow backend, '
               'you must first install TensorFlow!')
    raise ImportError(message) from error

import numpy as np

from . import Backend


class TensorflowBackend(Backend):
    backend_name = 'tensorflow'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=np.float32, device=None, device_id=None):
        if isinstance(data, tf.Tensor):
            return data

        out = tf.Variable(data, dtype=dtype)
        return out.gpu(device_id) if device == 'gpu' else out

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, tf.Tensor) or isinstance(tensor, tf.Variable)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        elif isinstance(tensor, tf.Variable):
            return tf.convert_to_tensor(tensor).numpy()
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
            a_min = tf.reduce_min(input_tensor=tensor_)

        if a_max is not None:
            a_max = self.tensor(a_max, **self.context(tensor_))
        else:
            a_max = tf.reduce_max(input_tensor=tensor_)

        return tf.clip_by_value(tensor_, clip_value_min=a_min, clip_value_max=a_max)

    def moveaxis(self, tensor, source, target):
        axes = list(range(self.ndim(tensor)))
        if source < 0: source = axes[source]
        if target < 0: target = axes[target]
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
        return tf.transpose(a=tensor, perm=axes)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        if order == 'inf':
            order = np.inf
        return tf.norm(tensor=tensor, ord=order, axis=axis)

    def dot(self, tensor1, tensor2):
        return tf.tensordot(tensor1, tensor2, axes=([self.ndim(tensor1) - 1], [0]))

    @staticmethod
    def conj(x, *args, **kwargs):
        """WARNING: IDENTITY FUNCTION (does nothing)

            This backend currently does not support complex tensors
        """
        return x

    @staticmethod
    def solve(lhs, rhs):
        squeeze = False
        if rhs.ndim == 1:
            squeeze = [-1]
            rhs = tf.reshape(rhs, (-1, 1))
        res = tf.linalg.solve(lhs, rhs)
        if squeeze:
            res = tf.squeeze(res, squeeze)
        return res

    @staticmethod
    def truncated_svd(matrix, n_eigenvecs=None, **kwargs):
        """Computes an SVD on `matrix`

        Parameters
        ----------
        matrix : 2D-array
        n_eigenvecs : int, optional, default is None
            if specified, number of eigen[vectors-values] to return
        **kwargs : optional
            kwargs are used to absorb the difference of parameters among the other SVD functions

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

        S, U, V = tf.linalg.svd(matrix, full_matrices=full_matrices)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], tf.transpose(a=V)[:n_eigenvecs, :]
        return U, S, V

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'truncated_svd': self.truncated_svd}
    
    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            direction = 'DESCENDING'
        else:
            direction = 'ASCENDING'
            
        if axis is None:
            tensor = tf.reshape(tensor, [-1])
            axis = -1

        return tf.sort(tensor, axis=axis, direction = direction)
    
    def index_update(self, tensor, indices, values):
        if not isinstance(tensor, tf.Variable):
            tensor = tf.Variable(tensor)
            to_tensor = True
        else:
            to_tensor = False
        
        if isinstance(values, int):
            values = tf.constant(np.ones(self.shape(tensor[indices]))*values,
                                 **self.context(tensor))
        
        tensor = tensor[indices].assign(values)

        if to_tensor:
            return tf.convert_to_tensor(tensor)
        else:
            return tensor


_FUN_NAMES = [
    # source_fun, target_fun
    (np.int32, 'int32'),
    (np.int64, 'int64'),
    (np.float32, 'float32'),
    (np.float64, 'float64'),
    (tf.ones, 'ones'),
    (tf.zeros, 'zeros'),
    (tf.linalg.tensor_diag, 'diag'),
    (tf.zeros_like, 'zeros_like'),
    (tf.eye, 'eye'),
    (tf.reshape, 'reshape'),
    (tf.transpose, 'transpose'),
    (tf.where, 'where'),
    (tf.sign, 'sign'),
    (tf.abs, 'abs'),
    (tf.sqrt, 'sqrt'),
    (tf.linalg.qr, 'qr'),
    #(tf.linalg.solve, 'solve'),
    (tf.argmin, 'argmin'),
    (tf.argmax, 'argmax'),
    (tf.stack, 'stack'),
    (tf.identity, 'copy'),
    (tf.concat, 'concatenate'),
    (tf.stack, 'stack'),
    (tf.reduce_min, 'min'),
    (tf.reduce_max, 'max'),
    (tf.reduce_mean, 'mean'),
    (tf.reduce_sum, 'sum'),
    (tf.reduce_prod, 'prod'),
    (tf.reduce_all, 'all'),
    (tf.einsum, 'einsum')
    ]
for source_fun, target_fun_name in _FUN_NAMES:
    TensorflowBackend.register_method(target_fun_name, source_fun)
del _FUN_NAMES

