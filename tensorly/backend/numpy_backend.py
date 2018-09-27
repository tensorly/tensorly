import warnings

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from . import _generics


# Author: Jean Kossaifi

# License: BSD 3 clause


backend = _generics.new_backend('numpy')


for name in ['int64', 'int32', 'float64', 'float32', 'reshape', 'moveaxis',
             'where', 'copy', 'transpose', 'arange', 'ones', 'zeros',
             'zeros_like', 'eye', 'dot', 'kron', 'concatenate', 'max', 'min',
             'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt']:
    backend.register(getattr(np, name), name)


backend.register(np.linalg.solve, name='solve')
backend.register(np.linalg.qr, name='qr')


@backend.register
def context(tensor):
    return {'dtype': tensor.dtype}


@backend.register
def tensor(data, dtype=None):
    return np.array(data, dtype=dtype)


@backend.register
def is_tensor(tensor):
    return isinstance(tensor, np.ndarray)


@backend.register
def to_numpy(tensor):
    return np.copy(tensor)


@backend.register
def assert_array_equal(a, b, **kwargs):
    return np.testing.assert_array_equal(a, b, **kwargs)


@backend.register
def assert_array_almost_equal(a, b, **kwargs):
    np.testing.assert_array_almost_equal(to_numpy(a), to_numpy(b), **kwargs)


backend.register(np.testing.assert_raises)
backend.register(np.testing.assert_equal)
backend.register(np.testing.assert_)


@backend.register
def shape(tensor):
    return tensor.shape


@backend.register
def ndim(tensor):
    return tensor.ndim


@backend.register
def clip(tensor, a_min=None, a_max=None, inplace=False):
    return np.clip(tensor, a_min, a_max)


@backend.register
def norm(tensor, order=2, axis=None):
    # handle difference in default axis notation
    if axis == ():
        axis = None

    if order == 'inf':
        return np.max(np.abs(tensor), axis=axis)
    if order == 1:
        return np.sum(np.abs(tensor), axis=axis)
    elif order == 2:
        return np.sqrt(np.sum(tensor**2, axis=axis))
    else:
        return np.sum(np.abs(tensor)**order, axis=axis)**(1/order)


@backend.register
def kr(matrices):
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))


@backend.register
def partial_svd(matrix, n_eigenvecs=None):
    # Check that matrix is... a matrix!
    if matrix.ndim != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
            matrix.ndim))

    # Choose what to do depending on the params
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
        max_dim = dim_2
    else:
        min_dim = dim_2
        max_dim = dim_1

    if n_eigenvecs >= min_dim:
        if n_eigenvecs > max_dim:
            warnings.warn(('Trying to compute SVD with n_eigenvecs={1}, which '
                           'is larger than max(matrix.shape)={1}. Setting '
                           'n_eigenvecs to {1}').format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim

        if n_eigenvecs is None or n_eigenvecs > min_dim:
            full_matrices = True
        else:
            full_matrices = False

        # Default on standard SVD
        U, S, V = scipy.linalg.svd(matrix, full_matrices=full_matrices)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        return U, S, V

    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(
                np.dot(matrix, matrix.T.conj()), k=n_eigenvecs, which='LM'
            )
            S = np.sqrt(S)
            V = np.dot(matrix.T.conj(), U * 1/S[None, :])
        else:
            S, V = scipy.sparse.linalg.eigsh(
                np.dot(matrix.T.conj(), matrix), k=n_eigenvecs, which='LM'
            )
            S = np.sqrt(S)
            U = np.dot(matrix, V) * 1/S[None, :]

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return U, S, V.T.conj()


SVD_FUNS = {'numpy_svd': partial_svd,
            'truncated_svd': partial_svd}
backend.register(SVD_FUNS, name='SVD_FUNS')
