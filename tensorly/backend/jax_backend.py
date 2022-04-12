import warnings
from distutils.version import LooseVersion

try:
    import jax
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
    import jax.scipy.special
except ImportError as error:
    message = ('Impossible to import Jax.\n'
               'To use TensorLy with the Jax backend, '
               'you must first install Jax!')
    raise ImportError(message) from error

import numpy
import copy

from .core import Backend


class JaxBackend(Backend, backend_name='jax'):

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, **kwargs):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return numpy.asarray(tensor)

    def copy(self, tensor):
        # See https://github.com/tensorly/tensorly/pull/397
        # and https://github.com/google/jax/issues/3473
        return self.tensor( tensor.copy(), **self.context(tensor))
        #return copy.copy(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def dot(a, b):
        return a.dot(b)

    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None, numpy_resid=True)
        return x, residuals

    def kr(self, matrices, weights=None, mask=None):
        n_columns = matrices[0].shape[1]
        n_factors = len(matrices)

        start = ord('a')
        common_dim = 'z'
        target = ''.join(chr(start + i) for i in range(n_factors))
        source = ','.join(i + common_dim for i in target)
        operation = source + '->' + target + common_dim

        if weights is not None:
            matrices = [m if i else m*self.reshape(weights, (1, -1)) for i, m in enumerate(matrices)]

        m = mask.reshape((-1, 1)) if mask is not None else 1
        return np.einsum(operation, *matrices).reshape((-1, n_columns))*m

    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            return np.flip(np.sort(tensor, axis=axis), axis = axis)
        else:
            return np.sort(tensor, axis=axis)

    @staticmethod
    def argsort(tensor, axis, descending = False):
        if descending:
            return np.argsort(-1 * tensor, axis=axis)
        else:
            return np.argsort(tensor, axis=axis)

for name in ['int64', 'int32', 'float64', 'float32', 'complex128', 'complex64', 
             'pi', 'e', 'inf', 'nan',
             'reshape', 'moveaxis',
             'where', 'transpose', 'arange', 'ones', 'zeros', 'flip', 'trace', 'any',
             'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min', 'matmul',
             'all', 'mean', 'sum', 'cumsum', 'count_nonzero', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'conj', 'diag', 'clip', 'einsum', 'log', 'log2', 'tensordot', 'exp',
             'sin', 'cos', 'tan', 
             'arcsin', 'arccos', 'arctan',
             'sinh', 'cosh', 'tanh', 
             'arcsinh', 'arccosh', 'arctanh',
            ]:
    JaxBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'svd', 'eigh']:
    JaxBackend.register_method(name, getattr(np.linalg, name))

if LooseVersion(jax.__version__) >= LooseVersion('0.3.0'):
    def index_update(tensor, indices, values):
        return tensor.at[indices].set(values)
    JaxBackend.register_method('index_update', index_update)
else:
    JaxBackend.register_method(name, getattr(jax.ops, name))

for name in ['digamma']:
    JaxBackend.register_method(name, getattr(jax.scipy.special, name))