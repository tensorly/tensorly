from copy import copy as _py_copy
from distutils.version import LooseVersion

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse.linalg
import sparse

from . import register_sparse_backend
from ....backend.core import Backend


_MIN_SPARSE_VERSION = '0.4.1+10.g81eccee'
if LooseVersion(sparse.__version__) < _MIN_SPARSE_VERSION:
    raise ImportError("numpy sparse backend requires `sparse` version >= %r"
                      % _MIN_SPARSE_VERSION)


def is_sparse(x):
    return isinstance(x, sparse.SparseArray)


class NumpySparseBackend(Backend):
    backend_name = 'numpy.sparse'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None):
        if is_sparse(data):
            return (data.astype(dtype)
                    if dtype is not None and dtype != data.dtype
                    else data)
        elif isinstance(data, np.ndarray):
            return sparse.COO.from_numpy(data.astype(dtype, copy=False))
        else:
            return sparse.COO.from_numpy(np.array(data, dtype=dtype))

    @staticmethod
    def is_tensor(obj):
        return is_sparse(obj)

    @staticmethod
    def to_numpy(tensor):
        return tensor.todense() if is_sparse(tensor) else np.array(tensor)

    @staticmethod
    def copy(tensor):
        return _py_copy(tensor)

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

    def dot(self, x, y):
        if is_sparse(x) or is_sparse(y):
            return sparse.dot(x, y)
        return np.dot(x, y)

    @staticmethod
    def partial_svd(matrix, n_eigenvecs=None):
        # Check that matrix is... a matrix!
        if matrix.ndim != 2:
            raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
                matrix.ndim))

        # Choose what to do depending on the params
        dim_1, dim_2 = matrix.shape
        if dim_1 <= dim_2:
            min_dim = dim_1
        else:
            min_dim = dim_2

        if not is_sparse(matrix) and (n_eigenvecs is None or n_eigenvecs >= min_dim):
            if n_eigenvecs is None or n_eigenvecs > min_dim:
                full_matrices = True
            else:
                full_matrices = False
            # Default on standard SVD
            U, S, V = scipy.linalg.svd(matrix, full_matrices=full_matrices)
            U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
            return U, S, V

        else:
            if n_eigenvecs is None:
                raise ValueError('n_eigenvecs cannot be none')
            if n_eigenvecs > min_dim:
                msg = ('n_eigenvecs={} if greater than the minimum matrix '
                       'dimension ({})')
                raise ValueError(msg.format(n_eigenvecs, min(matrix.shape)))
            if np.issubdtype(matrix.dtype, np.complexfloating):
                raise NotImplementedError("Complex dtypes")
            # We can perform a partial SVD
            # First choose whether to use X * X.T or X.T *X
            if dim_1 < dim_2:
                conj = matrix.T
                xxT = matrix.dot(conj)
                if is_sparse(xxT):
                    xxT = xxT.to_scipy_sparse()
                S, U = scipy.sparse.linalg.eigsh(xxT, k=n_eigenvecs, which='LM')
                S = np.sqrt(S)
                V = conj.dot(U / S[None, :])
            else:
                xTx = matrix.T.dot(matrix)
                if is_sparse(xTx):
                    xTx = xTx.to_scipy_sparse()
                S, V = scipy.sparse.linalg.eigsh(xTx, k=n_eigenvecs, which='LM')
                S = np.sqrt(S)
                U = matrix.dot(V / S[None, :])

            # WARNING: here, V is still the transpose of what it should be
            U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return U, S, V.T.conj()

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'truncated_svd': self.partial_svd}


for name in ['int64', 'int32', 'float64', 'float32', 'moveaxis', 'transpose',
             'reshape', 'ndim', 'shape', 'max', 'min', 'all', 'mean', 'sum',
             'prod', 'sqrt', 'abs', 'sign', 'clip', 'arange']:
    NumpySparseBackend.register_method(name, getattr(np, name))

for name in ['where', 'concatenate', 'kron', 'zeros', 'zeros_like', 'eye', 'ones']:
    NumpySparseBackend.register_method(name, getattr(sparse, name))


