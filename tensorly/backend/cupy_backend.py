"""
Core tensor operations with CuPy.
"""

# Author: Jean Kossaifi

# License: BSD 3 clause

# First check whether cupy is installed
try:
    import cupy as cp
except ImportError as error:
    message = ('Impossible to import cupy.\n'
               'To use TensorLy with the cupy backend, '
               'you must first install cupy!')
    raise ImportError(message) from error

import warnings
import numpy as np

from . import _generics
from .generic import kr, partial_svd


backend = _generics.new_backend('cupy')

for name in ['float64', 'float32', 'int64', 'int32', 'reshape', 'moveaxis',
             'transpose', 'copy', 'ones', 'zeros', 'zeros_like', 'eye',
             'arange', 'where', 'dot', 'kron', 'qr', 'concatenate', 'max',
             'min', 'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt']:
    backend.register(getattr(cp, name), name=name)

backend.register(kr)
backend.register(partial_svd)


@backend.register
def context(tensor):
    return {'dtype': tensor.dtype}


@backend.register
def tensor(data, dtype=cp.float32):
    return cp.array(data, dtype=dtype)


@backend.register
def is_tensor(tensor):
    return isinstance(tensor, cp.ndarray)


@backend.register
def to_numpy(tensor):
    if isinstance(tensor, cp.ndarray):
        return cp.asnumpy(tensor)
    return tensor


@backend.register
def shape(tensor):
    return tensor.shape


@backend.register
def ndim(tensor):
    return tensor.ndim


@backend.register
def clip(tensor, a_min=None, a_max=None):
    return cp.clip(tensor, a_min, a_max)


@backend.register
def norm(tensor, order=2, axis=None):
    # handle difference in default axis notation
    if axis == ():
        axis = None

    if order == 'inf':
        res = cp.max(cp.abs(tensor), axis=axis)
    elif order == 1:
        res = cp.sum(cp.abs(tensor), axis=axis)
    elif order == 2:
        res = cp.sqrt(cp.sum(tensor**2, axis=axis))
    else:
        res = cp.sum(cp.abs(tensor)**order, axis=axis)**(1 / order)

    if res.shape == ():
        return to_numpy(res)
    return res


@backend.register
def solve(matrix1, matrix2):
    try:
        cp.linalg.solve(matrix1, matrix2)
    except cp.cuda.cusolver.CUSOLVERError:
        warnings.warn('CuPy solver failed, using numpy.linalg.solve instead.')
        return tensor(np.linalg.solve(to_numpy(matrix1), to_numpy(matrix2)),
                      **context(matrix1))


def truncated_svd(matrix, n_eigenvecs=None):
    """Computes a truncated SVD on `matrix`

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

    U, S, V = cp.linalg.svd(tensor, full_matrices=full_matrices)
    U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
    return U, S, V


SVD_FUNS = {'numpy_svd': partial_svd,
            'truncated_svd': truncated_svd}
backend.register(SVD_FUNS, 'SVD_FUNS')


backend.register(np.testing.assert_raises)
backend.register(np.testing.assert_)


@backend.register
def assert_array_equal(a, b, **kwargs):
    np.testing.assert_array_equal(to_numpy(a), to_numpy(b), **kwargs)


@backend.register
def assert_array_almost_equal(a, b, decimal=4, **kwargs):
    np.testing.assert_array_almost_equal(to_numpy(a), to_numpy(b),
                                         decimal=decimal, **kwargs)


@backend.register
def assert_equal(actual, desired, err_msg='', verbose=True):
    if isinstance(actual, cp.ndarray):
        actual = to_numpy(actual)
        if actual.shape == (1, ):
            actual = actual[0]
    if isinstance(desired, cp.ndarray):
        desired = to_numpy(desired)
        if desired.shape == (1, ):
            desired = desired[0]
    np.testing.assert_equal(actual, desired)
