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
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)

    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return x, residuals

    def kr(self, matrices, weights=None, mask=None):
        n_columns = matrices[0].shape[1]
        n_factors = len(matrices)

        start = ord("a")
        common_dim = "z"
        target = "".join(chr(start + i) for i in range(n_factors))
        source = ",".join(i + common_dim for i in target)
        operation = source + "->" + target + common_dim

        if weights is not None:
            matrices = [
                m if i else m * self.reshape(weights, (1, -1))
                for i, m in enumerate(matrices)
            ]

        m = mask.reshape((-1, 1)) if mask is not None else 1
        return np.einsum(operation, *matrices).reshape((-1, n_columns)) * m


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "nan",
        "moveaxis",
        "trace",
        "copy",
        "transpose",
        "arange",
        "flip",
        "kron",
        "concatenate",
        "max",
        "mean",
        "sum",
        "argmin",
        "argmax",
        "sign",
        "stack",
        "conj",
        "diag",
        "log",
        "log2",
        "tensordot",
        "argsort",
        "sort",
        "dot",
        "shape",
    ]
):
    NumpyBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "svd", "eigh"]:
    NumpyBackend.register_method(name, getattr(np.linalg, name))

for name in ["digamma"]:
    NumpyBackend.register_method(name, getattr(scipy.special, name))
