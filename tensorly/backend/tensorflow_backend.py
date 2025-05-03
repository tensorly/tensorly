try:
    import tensorflow as tf
    import tensorflow.math as tfm
    import tensorflow.experimental.numpy as tnp
except ImportError as error:
    message = (
        "Impossible to import TensorFlow.\n"
        "To use TensorLy with the TensorFlow backend, "
        "you must first install TensorFlow!"
    )
    raise ImportError(message) from error

import numpy as np

from .core import Backend, backend_types, backend_basic_math, backend_array

tnp.experimental_enable_numpy_behavior()


class TensorflowBackend(Backend, backend_name="tensorflow"):
    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    # @staticmethod
    # def tensor(data, dtype=np.float64, device=None, device_id=None):
    #     if isinstance(data, tf.Tensor) or isinstance(data, tf.Variable):
    #         return tf.cast(data, dtype=dtype)
    #
    #     out = tf.Variable(data, dtype=dtype)
    #     return out.gpu(device_id) if device == "gpu" else out

    @staticmethod
    def tensor(data, dtype=None, device=None, device_id=None):
        # Determine the dtype and device from the input if not provided
        if dtype is None:
            if isinstance(data, tf.Tensor) or isinstance(data, tf.Variable):
                dtype = data.dtype
            else:  # for numpy arrays and lists
                dtype = tf.as_dtype(np.array(data).dtype)

        if device is None and device_id is None:
            if isinstance(data, tf.Tensor) or isinstance(data, tf.Variable):
                device = data.device

        # Create the tensor and cast to the determined dtype
        out = tnp.array(data, dtype=dtype)

        # If device or device_id is specified, place the tensor on the correct device
        if device is not None or device_id is not None:
            with tf.device(
                f"{device}:{device_id}" if device_id is not None else device
            ):
                out = tf.Variable(
                    out.numpy(), dtype=dtype
                )  # Re-wrap the tensor on the specified device

        return out

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, tf.Tensor) or isinstance(tensor, tf.Variable)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        elif isinstance(tensor, tf.Variable):
            return tf.convert_to_tensor(tensor).numpy()
        else:
            return tensor

    @staticmethod
    def shape(tensor):
        return tuple(tensor.shape.as_list())

    @staticmethod
    def norm(tensor, order=2, axis=None):
        if order == "inf":
            order = np.inf
        return tf.norm(tensor=tensor, ord=order, axis=axis)

    @staticmethod
    def solve(lhs, rhs):
        squeeze = False
        if tnp.ndim(rhs) == 1:
            squeeze = [-1]
            rhs = tf.reshape(rhs, (-1, 1))
        res = tf.linalg.solve(lhs, rhs)
        if squeeze:
            res = tf.squeeze(res, squeeze)
        return res

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return tnp.clip(tensor, a_min, a_max)

    @staticmethod
    def qr(tensor, mode="reduced"):
        if mode == "reduced":
            full_matrices = False
        elif mode == "complete":
            full_matrices = True

        return tf.linalg.qr(tensor, full_matrices=full_matrices)

    @staticmethod
    def lstsq(a, b):
        n = a.shape[1]
        if tf.rank(b) == 1:
            x = tf.squeeze(tf.linalg.lstsq(a, tf.expand_dims(b, -1), fast=False), -1)
        else:
            x = tf.linalg.lstsq(a, b, fast=False)
        residuals = tf.norm(tf.tensordot(a, x, 1) - b, axis=0) ** 2
        return x, residuals if tf.linalg.matrix_rank(a) == n else tf.constant([])

    def svd(self, matrix, full_matrices):
        """Correct for the atypical return order of tf.linalg.svd."""
        S, U, V = tf.linalg.svd(matrix, full_matrices=full_matrices)
        return U, S, tf.transpose(a=V)

    def index_update(self, tensor, indices, values):
        if not isinstance(tensor, tf.Variable):
            tensor = tf.Variable(tensor)
            to_tensor = True
        else:
            to_tensor = False

        if isinstance(values, int):
            values = tf.constant(
                np.ones(self.shape(tensor[indices])) * values, **self.context(tensor)
            )

        tensor = tensor[indices].assign(values)

        if to_tensor:
            return tf.convert_to_tensor(tensor)
        else:
            return tensor

    @staticmethod
    def logsumexp(tensor, axis=0):
        return tfm.reduce_logsumexp(tensor, axis=axis)


# Register numpy functions
for name in ["nan"]:
    TensorflowBackend.register_method(name, getattr(np, name))


# Register linalg functions
for name in ["diag", "eigh", "trace"]:
    TensorflowBackend.register_method(name, getattr(tf.linalg, name))


# Register tfm functions
TensorflowBackend.register_method("digamma", getattr(tfm, "digamma"))


# Register tnp functions
for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "log2",
        "concatenate",
        "flip",
        "dot",
        "argmin",
        "argmax",
        "conj",
        "tensordot",
        "stack",
        "copy",
        "max",
        "sign",
        "mean",
        "sum",
        "moveaxis",
        "ndim",
        "arange",
        "sort",
        "argsort",
        "flip",
        "stack",
        "transpose",
    ]
):
    TensorflowBackend.register_method(name, getattr(tnp, name))
