"""
Core tensor operations with PyTorch.
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

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
from torch import reshape, matmul as dot


from . import _generics
from .generic import kron, kr, partial_svd


backend = _generics.new_backend('pytorch')

for name in ['float64', 'float32', 'int64', 'int32', 'is_tensor', 'ones',
             'zeros', 'zeros_like', 'reshape', 'eye', 'max', 'min', 'prod',
             'abs', 'sqrt', 'sign', 'where', 'qr']:
    backend.register(getattr(torch, name), name=name)

backend.register(torch.matmul, name='dot')
backend.register(kron)
backend.register(kr)
backend.register(partial_svd)


@backend.register
def context(tensor):
    return {'dtype': tensor.dtype,
            'device': tensor.device,
            'requires_grad': tensor.requires_grad}


@backend.register
def tensor(data, dtype=torch.float32, device='cpu', requires_grad=False):
    if isinstance(data, np.ndarray):
        data = data.copy()
    return torch.tensor(data, dtype=dtype, device=device,
                        requires_grad=requires_grad)


@backend.register
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


@backend.register
def shape(tensor):
    return tensor.size()


@backend.register
def ndim(tensor):
    return tensor.dim()


@backend.register
def arange(start, stop=None, step=1.0):
    if stop is None:
        return torch.arange(start=0., end=float(start), step=float(step))
    else:
        return torch.arange(float(start), float(stop), float(step))


@backend.register
def clip(tensor, a_min=None, a_max=None, inplace=False):
    if a_max is None:
        a_max = torch.max(tensor)
    if a_min is None:
        a_min = torch.min(tensor)
    if inplace:
        return torch.clamp(tensor, a_min, a_max, out=tensor)
    else:
        return torch.clamp(tensor, a_min, a_max)


@backend.register
def all(tensor):
    return torch.sum(tensor != 0)


@backend.register
def transpose(tensor):
    axes = list(range(ndim(tensor)))[::-1]
    return tensor.permute(*axes)


@backend.register
def copy(tensor):
    return tensor.clone()


@backend.register
def moveaxis(tensor, source, target):
    axes = list(range(ndim(tensor)))
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


@backend.register
def solve(matrix1, matrix2):
    solution, _ = torch.gesv(matrix2, matrix1)
    return solution


@backend.register
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


@backend.register
def mean(tensor, axis=None):
    if axis is None:
        return torch.mean(tensor)
    else:
        return torch.mean(tensor, dim=axis)


@backend.register
def sum(tensor, axis=None):
    if axis is None:
        return torch.sum(tensor)
    else:
        return torch.sum(tensor, dim=axis)


@backend.register
def concatenate(tensors, axis=0):
    return torch.cat(tensors, dim=axis)


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


def truncated_svd(matrix, n_eigenvecs=None):
    """Computes a truncated SVD on `matrix` using pytorch's SVD

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    U : 2D-array
        of shape (matrix.shape[0], n_eigenvecs)
        contains the right singular vectors
    S : 1D-array
        of shape (n_eigenvecs, )
        contains the singular values of `matrix`
    V : 2D-array
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if n_eigenvecs is None or n_eigenvecs > min_dim:
        full_matrices = True
    else:
        full_matrices = False

    U, S, V = torch.svd(matrix, some=full_matrices)
    U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
    return U, S, V


def symeig_svd(matrix, n_eigenvecs=None):
    """Computes a truncated SVD on `matrix` using symeig

        Uses symeig on matrix.T.dot(matrix) or its transpose

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    U : 2D-array
        of shape (matrix.shape[0], n_eigenvecs)
        contains the right singular vectors
    S : 1D-array
        of shape (n_eigenvecs, )
        contains the singular values of `matrix`
    V : 2D-array
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    # Check that matrix is... a matrix!
    if ndim(matrix) != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is %d != 2'
                         % ndim(matrix))
    dim_1, dim_2 = shape(matrix)
    if dim_1 <= dim_2:
        min_dim = dim_1
        max_dim = dim_2
    else:
        min_dim = dim_2
        max_dim = dim_1

    if n_eigenvecs is None:
        n_eigenvecs = max_dim

    if min_dim <= n_eigenvecs:
        if n_eigenvecs > max_dim:
            warnings.warn('Trying to compute SVD with n_eigenvecs={0}, which '
                          'is larger than max(matrix.shape)={1}. Setting '
                          'n_eigenvecs to {1}'.format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim
        # we compute decomposition on the largest of the two to keep more eigenvecs
        dim_1, dim_2 = dim_2, dim_1

    if dim_1 < dim_2:
        S, U = torch.symeig(dot(matrix, transpose(matrix)))
        S = torch.sqrt(S)
        V = dot(transpose(matrix), U / reshape(S, (1, -1)))
    else:
        S, V = torch.symeig(dot(transpose(matrix), matrix))
        S = torch.sqrt(S)
        U = dot(matrix, V) / reshape(S, (1, -1))

    U, S, V = _reverse(U, 1), _reverse(S), _reverse(transpose(V), 0)
    return U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]


SVD_FUNS = {'numpy_svd': partial_svd,
            'truncated_svd': truncated_svd,
            'symeig_svd': symeig_svd}
backend.register(SVD_FUNS, 'SVD_FUNS')


backend.register(np.testing.assert_raises)
backend.register(np.testing.assert_)


@backend.register
def assert_array_equal(a, b, **kwargs):
    np.testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)


@backend.register
def assert_array_almost_equal(a, b, decimal=3, **kwargs):
    np.testing.assert_array_almost_equal(to_numpy(a), to_numpy(b),
                                         decimal=decimal, **kwargs)


@backend.register
def assert_equal(actual, desired, err_msg='', verbose=True):
    if isinstance(actual, torch.Tensor):
        actual = to_numpy(actual)
    if isinstance(desired, torch.Tensor):
        desired = to_numpy(desired)
    np.testing.assert_equal(actual, desired, err_msg=err_msg, verbose=verbose)
