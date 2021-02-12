try:
    import mxnet as mx
    from mxnet import numpy as np
except ImportError as error:
    message = ('Cannot import MXNet.\n'
               'To use TensorLy with the MXNet backend, '
               'you must first install MXNet!')
    raise ImportError(message) from error

import warnings
import numpy
from .core import Backend

mx.npx.set_np()


class MxnetBackend(Backend):
    backend_name = 'mxnet'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None):
        if dtype is None and isinstance(data, numpy.ndarray):
            dtype = data.dtype
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor.asnumpy()
        elif isinstance(tensor, numpy.ndarray):
            return tensor
        else:
            return numpy.array(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def dot(a, b):
        return np.dot(a, b)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == 'inf':
            return np.max(np.abs(tensor), axis=axis)
        if order == 1:
            return np.sum(np.abs(tensor), axis=axis)
        if order == 2:
            return np.sqrt(np.sum(tensor**2, axis=axis))
        
        return np.sum(np.abs(tensor)**order, axis=axis)**(1 / order)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)

    @staticmethod
    def conj(x, *args, **kwargs):
        """WARNING: IDENTITY FUNCTION (does nothing)

            This backend currently does not support complex tensors
        """
        return x

    def symeig_svd(self, matrix, n_eigenvecs=None, **kwargs):
        """Computes a truncated SVD on `matrix` using symeig

            Uses symeig on matrix.T.dot(matrix) or its transpose

        Parameters
        ----------
        matrix : 2D-array
        n_eigenvecs : int, optional, default is None
            if specified, number of eigen[vectors-values] to return
        **kwargs : optional
            kwargs are used to absorb the difference of parameters among the other SVD functions

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
        if self.ndim(matrix) != 2:
            raise ValueError('matrix be a matrix. matrix.ndim is %d != 2'
                             % self.ndim(matrix))

        dim_1, dim_2 = self.shape(matrix)
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
            S, U = np.linalg.eigh(np.dot(matrix, np.transpose(matrix)))
            S = self.sqrt(S)
            V = np.dot(np.transpose(matrix), U / np.reshape(S, (1, -1)))
        else:
            S, V = np.linalg.eigh(np.dot(np.transpose(matrix), matrix))
            S = self.sqrt(S)
            U = np.dot(matrix, V) / np.reshape(S, (1, -1))

        U, S, V = U[:, ::-1], S[::-1], np.transpose(V)[::-1, :]
        return U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'truncated_svd': self.truncated_svd,
                'symeig_svd': self.symeig_svd}
    
    @staticmethod
    def sort(tensor, axis, descending = False):
        if descending:
            return np.flip(np.sort(tensor, axis=axis), axis = axis)
        else:
            return np.sort(tensor, axis=axis)


for name in ['int64', 'int32', 'float64', 'float32', 'reshape', 'moveaxis',
             'where', 'copy', 'transpose', 'arange', 'ones', 'zeros',
             'zeros_like', 'eye', 'concatenate', 'max', 'min',
             'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
             'argmax', 'stack', 'diag', 'einsum']:
    MxnetBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'svd']:
    MxnetBackend.register_method(name, getattr(np.linalg, name))
