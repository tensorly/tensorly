from copy import copy as _py_copy
from distutils.version import LooseVersion

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse.linalg
import sparse

from . import register_sparse_backend
from ....backend.core import Backend


_MIN_SPARSE_VERSION = "0.4.1+10.g81eccee"
if LooseVersion(sparse.__version__) < _MIN_SPARSE_VERSION:
    raise ImportError(
        "numpy sparse backend requires `sparse` version >= %r" % _MIN_SPARSE_VERSION
    )


def is_sparse(x):
    return isinstance(x, sparse.SparseArray)


class NumpySparseBackend(Backend, backend_name="numpy.sparse"):
    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}  # , 'density':tensor.density}

    @staticmethod
    def tensor(data, dtype=None):
        if is_sparse(data):
            return (
                data.astype(dtype)
                if dtype is not None and dtype != data.dtype
                else data
            )
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
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == "inf":
            return np.max(np.abs(tensor), axis=axis)
        if order == 1:
            return np.sum(np.abs(tensor), axis=axis)
        elif order == 2:
            return np.sqrt(np.sum(tensor**2, axis=axis))
        else:
            return np.sum(np.abs(tensor) ** order, axis=axis) ** (1 / order)

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


for name in [
    "int64",
    "int32",
    "float64",
    "float32",
    "transpose",
    "moveaxis",
    "reshape",
    "ndim",
    "max",
    "min",
    "all",
    "mean",
    "sum",
    "prod",
    "sqrt",
    "abs",
    "sign",
    "arange",
    "conj",
    "shape",
]:
    NumpySparseBackend.register_method(name, getattr(np, name))

for name in [
    "where",
    "concatenate",
    "kron",
    "zeros",
    "zeros_like",
    "eye",
    "ones",
    "stack",
]:
    NumpySparseBackend.register_method(name, getattr(sparse, name))
