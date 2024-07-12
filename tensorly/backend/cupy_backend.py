try:
    import cupy as cp
    import cupyx.scipy.special

except ImportError as error:
    message = (
        "Impossible to import cupy.\n"
        "To use TensorLy with the cupy backend, "
        "you must first install cupy!"
    )
    raise ImportError(message) from error

import warnings

from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
)


class CupyBackend(Backend, backend_name="cupy"):
    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

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
    def logsumexp(tensor, axis=0):
        max_tensor = cp.max(tensor, axis=axis, keepdims=True)
        return cp.squeeze(
            cp.log(cp.sum(cp.exp(tensor - max_tensor), axis=axis, keepdims=True))
            + max_tensor,
            axis=axis,
        )


for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "moveaxis",
        "nan",
        "transpose",
        "copy",
        "trace",
        "arange",
        "dot",
        "kron",
        "concatenate",
        "max",
        "flip",
        "mean",
        "argmax",
        "sum",
        "stack",
        "sign",
        "conj",
        "diag",
        "tensordot",
        "log2",
        "argsort",
        "sort",
        "shape",
    ]
):
    CupyBackend.register_method(name, getattr(cp, name))

for name in ["svd", "qr", "eigh", "solve", "lstsq"]:
    CupyBackend.register_method(name, getattr(cp.linalg, name))

CupyBackend.register_method("gamma", cp.random.gamma)
