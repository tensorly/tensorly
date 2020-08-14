import numpy as np
import tensorflow as tf
from ....backend.core import Backend
from copy import copy as _py_copy


def is_sparse(x):
    return isinstance(x, tf.sparse.SparseTensor)


class TensorflowSparseBackend(Backend):
    backend_name = 'tensorflow'

    @staticmethod
    def tensor(data, dtype=np.float32, device=None, device_id=None):
        if isinstance(data, tf.sparse.SparseTensor):
            return data
        elif isinstance(data, tuple):
            if len(data) == 3:
                if isinstance(data[0], np.ndarray):
                    if isinstance(data[1], np.ndarray):
                        if len(data[0]) == len(data[1]):
                            return tf.sparse.SparseTensor(indices=data[0], values=data[1], dense_shape=data[2])


    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype.as_numpy_dtype}

    @staticmethod
    def shape(tensor):
        if isinstance(tensor, tf.sparse.SparseTensor):
            return np.asarray(tensor.shape.as_list())
        else:
            return tensor.shape

    @staticmethod
    def is_tensor(obj):
        return is_sparse(obj)

    @staticmethod
    def to_numpy(tensor):
        return tf.sparse.to_dense(tensor).numpy() if is_sparse(tensor) else np.array(tensor)

    @staticmethod
    def copy(tensor):
        return _py_copy(tensor)

    @staticmethod
    def ndim(tensor):
        """Return the number of dimensions of a tensor"""
        if isinstance(tensor, tf.sparse.SparseTensor):
            return tensor.shape.ndims
        else:
            return tensor.ndim
    @staticmethod
    def norm(tensor, order=2, axis=None):
        if axis == ():
            axis = None

        values = tensor.values.numpy()
        if order == 'inf':
            return np.max(values, axis=axis)
        if order == 1:
            return np.sum(values, axis=axis)
        elif order == 2:
            return np.sqrt(np.sum(values ** 2, axis=axis))
        else:
            return np.sum(values**order, axis=axis) ** (1 / order)

    @staticmethod
    def values(tensor):
        return tensor.values.numpy()

    @staticmethod
    def indices(tensor):
        return tensor.indices.numpy()

    @staticmethod
    def dot(a, b):
        #for numpy ndarrays, not yet implemented for tensorflow
        return a.dot(b)

for name in ['int64', 'int32', 'float64', 'float32', 'reshape', 'moveaxis',
             'where', 'copy', 'transpose', 'arange', 'ones', 'zeros',
             'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min',
             'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'conj', 'diag']:
    TensorflowSparseBackend.register_method(name, getattr(np, name))
