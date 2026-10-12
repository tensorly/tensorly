import numpy as np
from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
)
import scipy.special


class NumpyBackend(Backend, backend_name="numpy"):
    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, **kwargs):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return np.copy(tensor)

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def sort(x, axis=-1, descending=False, stable=True):
        out = np.sort(x, axis=axis)
        return np.flip(out, axis=axis) if descending else out

    @staticmethod
    def logsumexp(tensor, axis=0):
        return scipy.special.logsumexp(tensor, axis=axis)


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "nan",
        "shape",
        "moveaxis",
        "trace",
        "copy",
        "concat",
        "permute_dims",
        "clip",
        "arange",
        "flip",
        "kron",
        "max",
        "mean",
        "sum",
        "argmin",
        "argmax",
        "sign",
        "stack",
        "conj",
        "real",
        "diag",
        "log",
        "log2",
        "tensordot",
        "argsort",
    ]
):
    NumpyBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "svd", "eigh", "lstsq"]:
    NumpyBackend.register_method(name, getattr(np.linalg, name))

for name in ["digamma"]:
    NumpyBackend.register_method(name, getattr(scipy.special, name))
