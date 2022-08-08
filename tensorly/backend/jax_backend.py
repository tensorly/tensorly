import warnings
from distutils.version import LooseVersion

try:
    import jax
    from jax.config import config

    config.update("jax_enable_x64", True)
    import jax.numpy as np
    import jax.scipy.special
except ImportError as error:
    message = (
        "Impossible to import Jax.\n"
        "To use TensorLy with the Jax backend, "
        "you must first install Jax!"
    )
    raise ImportError(message) from error

import numpy
import copy

from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
)


class JaxBackend(Backend, backend_name="jax"):
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
        return numpy.asarray(tensor)

    def copy(self, tensor):
        # See https://github.com/tensorly/tensorly/pull/397
        # and https://github.com/google/jax/issues/3473
        return self.tensor(tensor.copy(), **self.context(tensor))
        # return copy.copy(tensor)

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def lstsq(a, b):
        x, residuals, _, _ = np.linalg.lstsq(a, b, rcond=None, numpy_resid=True)
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
        "transpose",
        "arange",
        "flip",
        "trace",
        "kron",
        "concatenate",
        "max",
        "mean",
        "sum",
        "argmin",
        "argmax",
        "stack",
        "sign",
        "conj",
        "diag",
        "clip",
        "log2",
        "tensordot",
        "argsort",
        "sort",
        "dot",
        "shape",
    ]
):
    JaxBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "svd", "eigh"]:
    JaxBackend.register_method(name, getattr(np.linalg, name))

if LooseVersion(jax.__version__) >= LooseVersion("0.3.0"):

    def index_update(tensor, indices, values):
        return tensor.at[indices].set(values)

    JaxBackend.register_method("index_update", index_update)
else:
    JaxBackend.register_method(name, getattr(jax.ops, name))

for name in ["gamma"]:
    JaxBackend.register_method(name, getattr(jax.random, name))
