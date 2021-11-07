try:
    import mxnet as mx
    from mxnet import numpy as np
except ImportError as error:
    message = ('Cannot import MXNet.\n'
               'To use TensorLy with the MXNet backend, '
               'you must first install MXNet!')
    raise ImportError(message) from error

import warnings
import numpy
from .core import Backend

mx.npx.set_np()


class MxnetBackend(Backend, backend_name='mxnet'):

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None):
        if dtype is None and isinstance(data, numpy.ndarray):
            dtype = data.dtype
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor.asnumpy()
        elif isinstance(tensor, numpy.ndarray):
            return tensor
        else:
            return numpy.array(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def dot(a, b):
        return np.dot(a, b)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)

    @staticmethod
    def conj(x, *args, **kwargs):
        """WARNING: IDENTITY FUNCTION (does nothing)

            This backend currently does not support complex tensors
        """
        return x

    def svd(self, X, full_matrices=True):
        # MXNet doesn't provide an option for full_matrices=True
        if full_matrices is True:
            ctx = self.context(X)
            X = self.to_numpy(X)

            if X.shape[0] > X.shape[1]:
                U, S, V = numpy.linalg.svd(X.T)

                U, S, V = V.T, S, U.T
            else:
                U, S, V = numpy.linalg.svd(X)

            U = self.tensor(U, **ctx)
            S = self.tensor(S, **ctx)
            V = self.tensor(V, **ctx)

            return U, S, V

        if X.shape[0] > X.shape[1]:
            U, S, V = np.linalg.svd(X.T)

            U, S, V = V.T, S, U.T
        else:
            U, S, V = np.linalg.svd(X)
        
        return U, S, V

    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return x, residuals

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

for name in ['int64', 'int32', 'float64', 'float32', 'reshape', 'moveaxis',
             'where', 'copy', 'transpose', 'arange', 'ones', 'zeros', 'trace', 'any',
             'zeros_like', 'eye', 'concatenate', 'max', 'min', 'flip', 'matmul',
             'all', 'mean', 'sum', 'cumsum', 'count_nonzero',  'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'diag', 'einsum', 'log2', 'tensordot', 'sin', 'cos']:
    MxnetBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'eigh']:
    MxnetBackend.register_method(name, getattr(np.linalg, name))
