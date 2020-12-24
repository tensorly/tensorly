
try:
    import jax
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
except ImportError as error:
    message = ('Impossible to import Jax.\n'
               'To use TensorLy with the Jax backend, '
               'you must first install Jax!')
    raise ImportError(message) from error

import numpy
import copy

from .core import Backend



class JaxBackend(Backend):
    backend_name = 'jax'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return numpy.asarray(tensor)

    @staticmethod
    def copy(tensor):
        return copy.copy(tensor)


    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == 'inf':
            return np.max(np.abs(tensor), axis=axis)
        if order == 1:
            return np.sum(np.abs(tensor), axis=axis)
        elif order == 2:
            return np.sqrt(np.sum(tensor**2, axis=axis))
        else:
            return np.sum(np.abs(tensor)**order, axis=axis)**(1 / order)

    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            return np.flip(np.sort(tensor, axis=axis), axis = axis)
        else:
            return np.sort(tensor, axis=axis)

for name in ['int64', 'int32', 'float64', 'float32', 'reshape', 'moveaxis',
             'where', 'dot', 'transpose', 'arange', 'ones', 'zeros',
             'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min', 'shape',
             'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'conj', 'diag', 'clip', 'einsum']:
    JaxBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'svd']:
    JaxBackend.register_method(name, getattr(np.linalg, name))

for name in ['index', 'index_update']:
    JaxBackend.register_method(name, getattr(jax.ops, name))

