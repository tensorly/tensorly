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
    def moveaxis(tensor, source, target):
        axes = list(range(tensor.dim()))
        if source < 0: source = axes[source]
        if target < 0: target = axes[target]
        try:
            axes.pop(source)
        except IndexError:
            raise ValueError('Source should be in 0 <= source < tensor.ndim, '
                             'got %d' % source)
        try:
            axes.insert(target, source)
        except IndexError:
            raise ValueError('Destination should be in 0 <= destination < '
                             'tensor.ndim, got %d' % target)
        return tensor.permute(*axes)

    def solve(self, matrix1, matrix2):
        """Solve a linear system of equation

        Notes
        -----

        Previously, this was implemented as follows::

            if self.ndim(matrix2) < 2:
                # Currently, gesv doesn't support vectors for matrix2
                # So we instead solve a least square problem...
                solution, _ = torch.gels(matrix2, matrix1)
            else:
                solution, _ = torch.gesv(matrix2, matrix1)
            return solution

        """
        if self.ndim(matrix2) < 2:
            # Currently, solve doesn't support vectors for matrix2
            solution, _ = torch.solve(matrix2.unsqueeze(1), matrix1)
        else:
            solution, _ = torch.solve(matrix2, matrix1)
        return solution

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
    def _reverse(tensor, axis=0):
        """Reverses the elements along the specified dimension

        Parameters
        ----------
        tensor : tl.tensor
        axis : int, default is 0
            axis along which to reverse the ordering of the elements

        Returns
        -------
        reversed_tensor : for a 1-D tensor, returns the equivalent of
                        tensor[::-1] in NumPy
        """
        indices = torch.arange(tensor.shape[axis] - 1, -1, -1, dtype=torch.int64)
        return tensor.index_select(axis, indices)

    @staticmethod
    def svd(matrix, full_matrices=True):
        """Computes the standard SVD."""
        return torch.svd(matrix, some=full_matrices)
    
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
             'abs', 'sqrt', 'sign', 'where', 'qr', 'conj', 'diag', 'finfo', 'einsum']:
    PyTorchBackend.register_method(name, getattr(torch, name))

PyTorchBackend.register_method('dot', torch.matmul)
