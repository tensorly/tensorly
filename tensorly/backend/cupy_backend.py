try:
    import cupy as cp
    import cupyx.scipy.special

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
    def tensor(data, dtype=cp.float32, **kwargs):
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
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return cp.clip(tensor, a_min, a_max)

    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = cp.linalg.lstsq(a, b, rcond=None)
        return x, residuals


for name in ['float64', 'float32', 'int64', 'int32', 'complex128', 'complex64',
 'reshape', 'moveaxis',
             'pi', 'e', 'inf', 'nan',
             'transpose', 'copy', 'ones', 'zeros', 'zeros_like', 'eye', 'trace', 'any',
             'arange', 'where', 'dot', 'kron', 'concatenate', 'max', 'flip', 'matmul',
             'min', 'all', 'mean', 'sum', 'cumsum', 'count_nonzero', 'prod', 'sign', 'abs', 'sqrt', 'stack',
             'conj', 'diag', 'einsum', 'tensordot', 
             'log', 'log2', 'exp',
             'sin', 'cos', 'tan', 
             'arcsin', 'arccos', 'arctan',
             'sinh', 'cosh', 'tanh', 'argsort', 'sort', 'shape',
             'arcsinh', 'arccosh', 'arctanh',
             ]:
    CupyBackend.register_method(name, getattr(cp, name))

for name in ['svd', 'qr', 'eigh', 'solve']:
    CupyBackend.register_method(name, getattr(cp.linalg, name))

CupyBackend.regsiter_method('gamma', cp.random.gamma)