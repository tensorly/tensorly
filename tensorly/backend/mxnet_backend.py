try:
    import mxnet as mx
    from mxnet import numpy as np
except ImportError as error:
    message = (
        "Cannot import MXNet.\n"
        "To use TensorLy with the MXNet backend, "
        "you must first install MXNet!"
    )
    raise ImportError(message) from error

import numpy
from .core import Backend, backend_basic_math, backend_array

mx.npx.set_np()


class MxnetBackend(Backend, backend_name="mxnet"):
    def __init__(name):
        message = (
            "The MXNet backend is deprecated and will be removed in future versions.\n"
            "Please transition to another backend."
        )
        DeprecationWarning(message)
        super().__init__()

    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, **kwargs):
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
    def ndim(tensor):
        return tensor.ndim

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
    def logsumexp(x, axis=0):
        max_x = np.max(x, axis=axis, keepdims=True)
        return np.squeeze(
            max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True)),
            axis=axis,
        )


for name in (
    backend_basic_math
    + backend_array
    + [
        "int64",
        "int32",
        "float64",
        "float32",
        "pi",
        "e",
        "inf",
        "nan",
        "moveaxis",
        "copy",
        "transpose",
        "arange",
        "trace",
        "concatenate",
        "max",
        "sign",
        "flip",
        "mean",
        "sum",
        "argmin",
        "argmax",
        "stack",
        "diag",
        "log2",
        "tensordot",
        "exp",
        "argsort",
        "sort",
        "dot",
        "shape",
    ]
):
    MxnetBackend.register_method(name, getattr(np, name))

for name in ["solve", "qr", "eigh", "lstsq"]:
    MxnetBackend.register_method(name, getattr(np.linalg, name))
