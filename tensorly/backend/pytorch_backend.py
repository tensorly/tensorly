from packaging.version import Version

try:
    import torch
except ImportError as error:
    message = (
        "Impossible to import PyTorch.\n"
        "To use TensorLy with the PyTorch backend, "
        "you must first install PyTorch!"
    )
    raise ImportError(message) from error

import numpy as np

from .core import (
    Backend,
    backend_types,
    backend_basic_math,
    backend_array,
)

if Version(torch.__version__) < Version("2.0.0"):
    raise RuntimeError("TensorLy only supports pytorch v2.0.0 and above.")


def _all_dims(x, axis):
    return tuple(range(x.ndim)) if axis is None else axis


def _reduce_axes(fn, x, axis, keepdims):
    """Apply a single-axis reduction `fn` over one, several or all axes."""
    if x.ndim == 0:
        return fn(x)
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    for ax in sorted((a % x.ndim for a in axis), reverse=True):
        x = fn(x, dim=ax, keepdim=keepdims)
    return x


class PyTorchBackend(Backend, backend_name="pytorch"):
    @staticmethod
    def context(tensor):
        return {
            "dtype": tensor.dtype,
            "device": tensor.device,
            "requires_grad": tensor.requires_grad,
        }

    @staticmethod
    def tensor(data, dtype=None, device=None, requires_grad=None):
        """
        Tensor constructor for the PyTorch backend.

        Parameters
        ----------
        data : array-like
            Data for the tensor.
        dtype : torch.dtype, optional
            Data type of the tensor. If None, the dtype is inferred from the data.
        device : Union[str, torch.device], optional
            Device on which the tensor is allocated. If None, the device is inferred from the data in case of a torch
            Tensor.
        requires_grad : bool, optional.
            If autograd should record operations on the returned tensor. If None, requires_grad is inferred from the
            data.

        """
        if isinstance(data, torch.Tensor):
            # If source is a tensor, use clone-detach as suggested by PyTorch
            tensor = data.clone().detach()
        else:
            # Else, use PyTorch's tensor constructor
            tensor = torch.tensor(data)

        # Set dtype/device/requires_grad if specified
        if dtype is not None:
            tensor = tensor.type(dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        if requires_grad is not None:
            tensor.requires_grad_(requires_grad)
        return tensor

    @staticmethod
    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.cuda:
                tensor = tensor.cpu()
            return tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.asarray(tensor)

    @staticmethod
    def shape(tensor):
        return tuple(tensor.shape)

    @staticmethod
    def ndim(tensor):
        return tensor.dim()

    @staticmethod
    def arange(start, stop=None, step=1, *args, **kwargs):
        if stop is None:
            return torch.arange(start=0, end=start, step=step, *args, **kwargs)
        else:
            return torch.arange(start, stop, step, *args, **kwargs)

    @staticmethod
    def clip(x, min=None, max=None, inplace=False):
        if inplace:
            return torch.clip(x, min, max, out=x)
        else:
            return torch.clip(x, min, max)

    @staticmethod
    def permute_dims(x, axes):
        return x.permute(*axes)

    @staticmethod
    def copy(tensor):
        return tensor.clone()

    @staticmethod
    def norm(tensor, order=None, axis=None):
        # pytorch does not accept `None` for any keyword arguments. additionally,
        # pytorch doesn't seems to support keyword arguments in the first place
        kwds = {}
        if axis is not None:
            kwds["dim"] = axis
        if order and order != "inf":
            kwds["p"] = order

        if order == "inf":
            res = torch.max(torch.abs(tensor), **kwds)
            if axis is not None:
                return res[0]  # ignore indices output
            return res
        return torch.norm(tensor, **kwds)

    @staticmethod
    def flip(tensor, axis=None):
        if isinstance(axis, int):
            axis = [axis]

        if axis is None:
            return torch.flip(tensor, dims=[i for i in range(tensor.ndim)])
        else:
            return torch.flip(tensor, dims=axis)

    @staticmethod
    def argmin(input, axis=None):
        return torch.argmin(input, dim=axis)

    @staticmethod
    def argmax(input, axis=None):
        return torch.argmax(input, dim=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def diag(tensor, k=0):
        return torch.diag(tensor, diagonal=k)

    @staticmethod
    def max(x, /, *, axis=None, keepdims=False):
        return torch.amax(x, dim=_all_dims(x, axis), keepdim=keepdims)

    @staticmethod
    def min(x, /, *, axis=None, keepdims=False):
        return torch.amin(x, dim=_all_dims(x, axis), keepdim=keepdims)

    @staticmethod
    def mean(x, /, *, axis=None, keepdims=False):
        return torch.mean(x, dim=_all_dims(x, axis), keepdim=keepdims)

    @staticmethod
    def sum(x, /, *, axis=None, keepdims=False):
        return torch.sum(x, dim=_all_dims(x, axis), keepdim=keepdims)

    @staticmethod
    def prod(x, /, *, axis=None, keepdims=False):
        return _reduce_axes(torch.prod, x, axis, keepdims)

    @staticmethod
    def all(x, /, *, axis=None, keepdims=False):
        return _reduce_axes(torch.all, x, axis, keepdims)

    @staticmethod
    def any(x, /, *, axis=None, keepdims=False):
        return _reduce_axes(torch.any, x, axis, keepdims)

    @staticmethod
    def tensordot(x1, x2, /, *, axes=2):
        return torch.tensordot(x1, x2, dims=axes)

    @staticmethod
    def eye(
        n_rows, n_cols=None, /, *, k=0, dtype=None, device=None, requires_grad=False
    ):
        n_cols = n_rows if n_cols is None else n_cols
        if k == 0:
            return torch.eye(
                n_rows,
                n_cols,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        rows = torch.arange(n_rows, device=device)[:, None]
        cols = torch.arange(n_cols, device=device)[None, :]
        out = (cols - rows == k).to(dtype or torch.get_default_dtype())
        return out.requires_grad_(requires_grad)

    @staticmethod
    def trace(x, /, *, offset=0):
        return torch.diagonal(x, offset=offset, dim1=-2, dim2=-1).sum(-1)

    @staticmethod
    def argsort(x, axis=-1, descending=False, stable=True):
        return torch.argsort(x, dim=axis, descending=descending, stable=stable)

    @staticmethod
    def sort(x, axis=-1, descending=False, stable=True):
        return torch.sort(x, dim=axis, descending=descending, stable=stable).values

    @staticmethod
    def lstsq(a, b, rcond=None, driver="gelsd"):
        return torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)

    @staticmethod
    def sign(tensor):
        """torch.sign does not support complex numbers."""
        return torch.sgn(tensor)

    @staticmethod
    def logsumexp(tensor, axis=0):
        return torch.logsumexp(tensor, dim=axis)


# Register the other functions
for name in (
    backend_types
    + backend_basic_math
    + backend_array
    + [
        "nan",
        "is_tensor",
        "conj",
        "real",
        "finfo",
        "log2",
        "digamma",
        "concat",
    ]
):
    if name in PyTorchBackend.__dict__:
        continue  # implemented in the class above
    PyTorchBackend.register_method(name, getattr(torch, name))


for name in ["kron", "moveaxis"]:
    PyTorchBackend.register_method(name, getattr(torch, name))

for name in ["solve", "qr", "svd", "eigh"]:
    PyTorchBackend.register_method(name, getattr(torch.linalg, name))
