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

    # moveaxis and shape are temporarily redefine to fix issue #131
    # Using the builting functionsn raises a TypeError: 
    #     no implementation found for 'numpy.shape' on types 
    #     that implement __array_function__: [<class 'sparse._coo.core.COO'>]
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
        return self.transpose(tensor, axes)

    # Temporary, see moveaxis above
    @staticmethod
    def shape(tensor):
        return tensor.shape

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
    def clip(tensor, a_min=None, a_max=None, inplace=False):
        return np.clip(tensor, a_min, a_max)

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

    def solve(self, A, b):
        """
        Compute x s.t. Ax = b
        """
        if is_sparse(A) or is_sparse(b):
            A, b = A.tocsc(), b.tocsc()
            x = sparse.COO(scipy.sparse.linalg.spsolve(A, b))
        else:
            x = np.linalg.solve(A, b)

        return x

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
        elif n_eigenvecs is None:
            raise ValueError('n_eigenvecs cannot be none')
        elif is_sparse(matrix) and matrix.nnz == 0:
            # all-zeros matrix, so we should do a quick return.
            U = sparse.eye(dim_1, n_eigenvecs, dtype=matrix.dtype)
            S = np.zeros(n_eigenvecs, dtype=matrix.dtype)
            V = sparse.eye(dim_2, n_eigenvecs, dtype=matrix.dtype)
        else:
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
                if n_eigenvecs >= xxT.shape[0]:
                    # use dense form when sparse form will fail
                    S, U = scipy.linalg.eigh(xxT.toarray())
                else:
                    S, U = scipy.sparse.linalg.eigsh(xxT, k=n_eigenvecs, which='LM')
                S = np.sqrt(S)
                V = conj.dot(U / S[None, :])
            else:
                xTx = matrix.T.dot(matrix)
                if is_sparse(xTx):
                    xTx = xTx.to_scipy_sparse()
                if n_eigenvecs >= xTx.shape[0]:
                    # use dense form when sparse form will fail
                    S, V = scipy.linalg.eigh(xTx.toarray())
                else:
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


for name in ['int64', 'int32', 'float64', 'float32', 'transpose',
             'reshape', 'ndim', 'max', 'min', 'all', 'mean', 'sum',
             'prod', 'sqrt', 'abs', 'sign', 'clip', 'arange', 'conj']:
    NumpySparseBackend.register_method(name, getattr(np, name))

for name in ['where', 'concatenate', 'kron', 'zeros', 'zeros_like', 'eye',
             'ones', 'stack']:
    NumpySparseBackend.register_method(name, getattr(sparse, name))


