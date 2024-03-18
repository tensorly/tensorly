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


if Version(torch.__version__) < Version("1.9.0"):
    raise RuntimeError("TensorLy only supports pytorch v1.9.0 and above.")


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
    def arange(start, stop=None, step=1.0, *args, **kwargs):
        if stop is None:
            return torch.arange(
                start=0.0, end=float(start), step=float(step), *args, **kwargs
            )
        else:
            return torch.arange(float(start), float(stop), float(step), *args, **kwargs)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None, inplace=False):
        if inplace:
            return torch.clip(tensor, a_min, a_max, out=tensor)
        else:
            return torch.clip(tensor, a_min, a_max)

    @staticmethod
    def all(tensor):
        return torch.sum(tensor != 0)

    def transpose(self, tensor, axes=None):
        axes = axes or list(range(self.ndim(tensor)))[::-1]
        return tensor.permute(*axes)

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
    def dot(a, b):
        if a.ndim > 2 and b.ndim > 2:
            return torch.tensordot(a, b, dims=([-1], [-2]))
        if not a.ndim or not b.ndim:
            return a * b
        return torch.matmul(a, b)

    @staticmethod
    def tensordot(a, b, axes=2, **kwargs):
        return torch.tensordot(a, b, dims=axes, **kwargs)

    @staticmethod
    def mean(tensor, axis=None):
        if axis is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim=axis)

    @staticmethod
    def sum(tensor, axis=None, keepdims=False):
        if axis is None:
            axis = tuple(range(tensor.ndim))
        return torch.sum(tensor, dim=axis, keepdim=keepdims)

    @staticmethod
    def max(tensor, axis=None):
        if axis is None:
            return torch.max(tensor)
        else:
            return torch.max(tensor, dim=axis)[0]

    @staticmethod
    def flip(tensor, axis=None):
        if isinstance(axis, int):
            axis = [axis]

        if axis is None:
            return torch.flip(tensor, dims=[i for i in range(tensor.ndim)])
        else:
            return torch.flip(tensor, dims=axis)

    @staticmethod
    def concatenate(tensors, axis=0):
        return torch.cat(tensors, dim=axis)

    @staticmethod
    def argmin(input, axis=None):
        return torch.argmin(input, dim=axis)

    @staticmethod
    def argsort(input, axis=None):
        return torch.argsort(input, dim=axis)

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
    def sort(tensor, axis):
        if axis is None:
            tensor = tensor.flatten()
            axis = -1

        return torch.sort(tensor, dim=axis).values

    @staticmethod
    def update_index(tensor, index, values):
        tensor.index_put_(index, values)

    @staticmethod
    def lstsq(a, b, rcond=None, driver="gelsd"):
        return torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)

    @staticmethod
    def eigh(tensor):
        """Legacy only, deprecated from PyTorch 1.8.0"""
        return torch.symeig(tensor, eigenvectors=True)

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
        "trace",
        "conj",
        "finfo",
        "log2",
        "digamma",
    ]
):
    PyTorchBackend.register_method(name, getattr(torch, name))


for name in ["kron", "moveaxis"]:
    PyTorchBackend.register_method(name, getattr(torch, name))

for name in ["solve", "qr", "svd", "eigh"]:
    PyTorchBackend.register_method(name, getattr(torch.linalg, name))
