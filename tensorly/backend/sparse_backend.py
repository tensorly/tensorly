import numpy as np
import scipy.linalg
import scipy.sparse.linalg

# TODO: rehape, moveaxis, where, copy, transpose
# TODO: arange, ones, zeros, zeros_like
# TODO: kron
# TODO: solve, qr
from numpy import reshape, moveaxis, copy
from numpy import arange, ones, zeros, zeros_like
from numpy import kron
from numpy import max, min, maximum, all, mean, sum, sign, abs, prod, sqrt
from numpy.linalg import solve, qr
import opt_einsum

import sparse
from .. import numpy_backend

from sparse import concatenate, where


assert_raises = np.testing.assert_raises
assert_equal = np.testing.assert_equal
assert_ = np.testing.assert_
assert_array_equal = np.testing.assert_array_equal

tensor = numpy_backend.tensor
shape = numpy_backend.shape
ndim = numpy_backend.ndim
clip = numpy_backend.clip
norm = numpy_backend.norm
assert_array_almost_equal = numpy_backend.assert_array_almost_equal


def dot(a, b):
    res = sparse.dot(a, b)
    return tensor(res)


def transpose(x):
    return x.T


def context(tensor):
    return {'dtype': tensor.dtype, 'coords': tensor.coords}


def tensor(data, dtype=None, coords=None):
    if isinstance(data, sparse.SparseArray):
        if dtype is not None:
            data = data.astype(dtype)
        return data
    data = np.asanyarray(data)
    if dtype is not None:
        data = data.astype(dtype)
    if isinstance(data, np.ndarray):
        if data.dtype.kind == 'f':
            i = data < np.finfo(data.dtype).resolution
            data[i] = 0
        return sparse.COO.from_numpy(data)
    assert coords is not None
    return sparse.COO(coords, data)


def to_numpy(coo_tensor):
    return coo_tensor.todense()


def kr(matrices):
    raise NotImplementedError


def partial_svd(matrix, n_eigenvecs=None):
    if n_eigenvecs is None:
        n_eigenvecs = min(6, *matrix.shape)  # 6 is the default for eigsh
        if min(matrix.shape) == n_eigenvecs:
            n_eigenvecs -= 1

    U, S, V = numpy_backend.partial_svd(matrix, n_eigenvecs=n_eigenvecs,
                                        sparse=True)

    return U, S, V
