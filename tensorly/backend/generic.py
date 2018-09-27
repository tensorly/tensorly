from .. import backend as T
from . import numpy_backend


def kron(matrix1, matrix2):
    """A generic implementation of kron"""
    s1, s2 = T.shape(matrix1)
    s3, s4 = T.shape(matrix2)
    a = T.reshape(matrix1, (s1, 1, s2, 1))
    b = T.reshape(matrix2, (1, s3, 1, s4))
    return T.reshape(a * b, (s1 * s3, s2 * s4))


def kr(matrices):
    """A generic implementation of kr"""
    if len(matrices) < 2:
        raise ValueError('kr requires a list of at least 2 matrices, but {} '
                         'given.'.format(len(matrices)))

    n_col = T.shape(matrices[0])[1]
    for i, e in enumerate(matrices[1:]):
        if not i:
            res = matrices[0]
        s1, s2 = T.shape(res)
        s3, s4 = T.shape(e)
        if not s2 == s4 == n_col:
            raise ValueError('All matrices should have the same number of columns.')
        a = T.reshape(res, (s1, 1, s2))
        b = T.reshape(e, (1, s3, s4))
        res = T.reshape(a * b, (-1, n_col))
    return res


def partial_svd(matrix, n_eigenvecs=None):
    ctx = T.context(matrix)
    matrix = T.to_numpy(matrix)
    U, S, V = numpy_backend.partial_svd(matrix, n_eigenvecs)
    return T.tensor(U, **ctx), T.tensor(S, **ctx), T.tensor(V, **ctx)
