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


class MxnetBackend(Backend):
    backend_name = 'mxnet'

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
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == 'inf':
            return np.max(np.abs(tensor), axis=axis)
        if order == 1:
            return np.sum(np.abs(tensor), axis=axis)
        if order == 2:
            return np.sqrt(np.sum(tensor**2, axis=axis))
        
        return np.sum(np.abs(tensor)**order, axis=axis)**(1 / order)

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
    def sort(tensor, axis, descending = False):
        if descending:
            return np.flip(np.sort(tensor, axis=axis), axis = axis)
        else:
            return np.sort(tensor, axis=axis)


for name in ['int64', 'int32', 'float64', 'float32', 'reshape', 'moveaxis',
             'where', 'copy', 'transpose', 'arange', 'ones', 'zeros',
             'zeros_like', 'eye', 'concatenate', 'max', 'min',
             'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'diag', 'einsum']:
    MxnetBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'eigh']:
    MxnetBackend.register_method(name, getattr(np.linalg, name))
