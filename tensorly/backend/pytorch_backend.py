import warnings
from distutils.version import LooseVersion

try:
    import torch
except ImportError as error:
    message = ('Impossible to import PyTorch.\n'
               'To use TensorLy with the PyTorch backend, '
               'you must first install PyTorch!')
    raise ImportError(message) from error

if LooseVersion(torch.__version__) < LooseVersion('0.4.0'):
    raise ImportError('You are using version=%r of PyTorch.'
                      'Please update to "0.4.0" or higher.'
                      % torch.__version__)

import numpy as np

from .core import Backend


class PyTorchBackend(Backend):
    backend_name = 'pytorch'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype,
                'device': tensor.device,
                'requires_grad': tensor.requires_grad}

    @staticmethod
    def tensor(data, dtype=torch.float32, device='cpu', requires_grad=False):
        if isinstance(data, np.ndarray):
            data = data.copy()
        return torch.tensor(data, dtype=dtype, device=device,
                            requires_grad=requires_grad)

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
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.dim()

    @staticmethod
    def arange(start, stop=None, step=1.0, *args, **kwargs):
        if stop is None:
            return torch.arange(start=0., end=float(start), step=float(step), *args, **kwargs)
        else:
            return torch.arange(float(start), float(stop), float(step), *args, **kwargs)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None, inplace=False):
        if a_max is None:
            a_max = torch.max(tensor)
        if a_min is None:
            a_min = torch.min(tensor)
        if inplace:
            return torch.clamp(tensor, a_min, a_max, out=tensor)
        else:
            return torch.clamp(tensor, a_min, a_max)

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
    def norm(tensor, order=2, axis=None):
        # pytorch does not accept `None` for any keyword arguments. additionally,
        # pytorch doesn't seems to support keyword arguments in the first place
        kwds = {}
        if axis is not None:
            kwds['dim'] = axis
        if order and order != 'inf':
            kwds['p'] = order

        if order == 'inf':
            res = torch.max(torch.abs(tensor), **kwds)
            if axis is not None:
                return res[0]  # ignore indices output
            return res
        return torch.norm(tensor, **kwds)

    @staticmethod
    def mean(tensor, axis=None):
        if axis is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim=axis)

    @staticmethod
    def sum(tensor, axis=None):
        if axis is None:
            return torch.sum(tensor)
        else:
            return torch.sum(tensor, dim=axis)
    
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
    def argmax(input, axis=None):
            return torch.argmax(input, dim=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return torch.stack(arrays, dim=axis)
    
    @staticmethod
    def sort(tensor, axis, descending = False):
        if axis is None:
            tensor = tensor.flatten()
            axis = -1

        return torch.sort(tensor, dim=axis, descending = descending).values

    @staticmethod
    def update_index(tensor, index, values):
        tensor.index_put_(index, values)

for name in ['float64', 'float32', 'int64', 'int32', 'is_tensor', 'ones',
             'zeros', 'zeros_like', 'reshape', 'eye', 'max', 'min', 'prod',
             'abs', 'sqrt', 'sign', 'where', 'qr', 'conj', 'diag', 'finfo', 'einsum', 'log2']:

    PyTorchBackend.register_method(name, getattr(torch, name))

for name in ['solve', 'qr', 'svd', 'eigh']:
    PyTorchBackend.register_method(name, getattr(torch.linalg, name))

PyTorchBackend.register_method('dot', torch.matmul)
