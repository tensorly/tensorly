try:
    import cupy as cp
except ImportError as error:
    message = ('Impossible to import cupy.\n'
               'To use TensorLy with the cupy backend, '
               'you must first install cupy!')
    raise ImportError(message) from error

import warnings
import numpy as np

from .core import Backend


class CupyBackend(Backend, backend_name='cupy'):

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=cp.float32):
        return cp.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, cp.ndarray)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, cp.ndarray):
            return cp.asnumpy(tensor)
        return tensor

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return cp.clip(tensor, a_min, a_max)

    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = cp.linalg.lstsq(a, b, rcond=None)
        return x, residuals

    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            return cp.flip(cp.sort(tensor, axis=axis), axis = axis)
        else:
            return cp.sort(tensor, axis=axis)

    @staticmethod
    def argsort(tensor, axis, descending = False):
        if descending:
            return np.argsort(-1 * tensor, axis=axis)
        else:
            return np.argsort(tensor, axis=axis)


for name in ['float64', 'float32', 'int64', 'int32', 'complex128', 'complex64', 'reshape', 'moveaxis',
             'transpose', 'copy', 'ones', 'zeros', 'zeros_like', 'eye', 'trace', 'any',
             'arange', 'where', 'dot', 'kron', 'concatenate', 'max', 'flip', 'matmul',
             'min', 'all', 'mean', 'sum', 'cumsum', 'count_nonzero', 'prod', 'sign', 'abs', 'sqrt', 'stack',
             'conj', 'diag', 'einsum', 'log2', 'tensordot']:
    CupyBackend.register_method(name, getattr(cp, name))

for name in ['svd', 'qr', 'eigh', 'solve']:
    CupyBackend.register_method(name, getattr(cp.linalg, name))
