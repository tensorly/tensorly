try:
    import mxnet as mx
except ImportError as error:
    message = ('Impossible to import MXNet.\n'
               'To use TensorLy with the MXNet backend, '
               'you must first install MXNet!')
    raise ImportError(message) from error

import math
import warnings

import numpy
from mxnet import nd
from mxnet.ndarray import reshape, dot, transpose, stack

from .core import Backend


class MxnetBackend(Backend):
    backend_name = 'mxnet'

    @staticmethod
    def context(tensor):
        return {'ctx': tensor.context, 'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, ctx=mx.cpu(), dtype=numpy.float32):
        if dtype is None and isinstance(data, numpy.ndarray):
            dtype = data.dtype
        return nd.array(data, ctx=ctx, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, nd.NDArray)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, nd.NDArray):
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
    def reshape(tensor, shape):
        if not shape:
            shape = [1]
        return nd.reshape(tensor, shape)

    def solve(self, matrix1, matrix2):
        ctx = self.context(matrix1)
        matrix1 = self.to_numpy(matrix1)
        matrix2 = self.to_numpy(matrix2)
        res = numpy.linalg.solve(matrix1, matrix2)
        return self.tensor(res, **ctx)

    @staticmethod
    def min(tensor, *args, **kwargs):
        if isinstance(tensor, nd.NDArray):
            return nd.min(tensor, *args, **kwargs).asscalar()
        else:
            return numpy.min(tensor, *args, **kwargs)

    @staticmethod
    def max(tensor, *args, **kwargs):
        if isinstance(tensor, nd.NDArray):
            return nd.max(tensor, *args, **kwargs).asscalar()
        else:
            return numpy.max(tensor, *args, **kwargs)

    @staticmethod
    def argmax(data=None, axis=None):
        res = nd.argmax(data, axis)
        if res.shape == (1,):
            return res.astype('int32').asscalar()
        else:
            return res

    @staticmethod
    def argmin(data=None, axis=None):
        res = nd.argmin(data, axis)
        if res.shape == (1,):
            return res.astype('int32').asscalar()
        else:
            return res

    @staticmethod
    def abs(tensor, **kwargs):
        if isinstance(tensor, nd.NDArray):
            return nd.abs(tensor, **kwargs)
        else:
            return numpy.abs(tensor, **kwargs)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis is None:
            axis = ()

        if order == 'inf':
            res = nd.max(nd.abs(tensor), axis=axis)
        elif order == 1:
            res = nd.sum(nd.abs(tensor), axis=axis)
        elif order == 2:
            res = nd.sqrt(nd.sum(tensor**2, axis=axis))
        else:
            res = nd.sum(nd.abs(tensor)**order, axis=axis)**(1 / order)

        if res.shape == (1,):
            return res.asscalar()

        return res

    def qr(self, matrix):
        # TODO: FIX THIS CASE
        # s1, s2 = matrix.shape
        # if s2 > s1:
        #     Q, L = nd.linalg.gelqf(matrix[:, :s1].T)
        #     return Q.T, L.T
        try:
            # NOTE - should be replaced with geqrf when available
            Q, L = nd.linalg.gelqf(matrix.T)
            return Q.T, L.T
        except:
            warnings.warn('This version of MXNet does not include the linear '
                          'algebra function gelqf(). Substituting with numpy.')
            ctx = self.context(matrix)
            Q, R = numpy.linalg.qr(self.to_numpy(matrix))
            return self.tensor(Q, **ctx), self.tensor(R, **ctx)

    @staticmethod
    def clip(tensor, a_min=None, a_max=None, indlace=False):
        if a_min is not None and a_max is not None:
            if indlace:
                nd.max(nd.min(tensor, a_max, out=tensor), a_min, out=tensor)
            else:
                tensor = nd.maximum(nd.minimum(tensor, a_max), a_min)
        elif min is not None:
            if indlace:
                nd.max(tensor, a_min, out=tensor)
            else:
                tensor = nd.maximum(tensor, a_min)
        elif max is not None:
            if indlace:
                nd.min(tensor, a_max, out=tensor)
            else:
                tensor = nd.minimum(tensor, a_max)
        return tensor

    @staticmethod
    def all(tensor):
        return nd.sum(tensor != 0).asscalar()

    @staticmethod
    def conj(x, *args, **kwargs):
        """WARNING: IDENTITY FUNCTION (does nothing)

            This backend currently does not support complex tensors
        """
        return x

    def moveaxis(self, tensor, source, target):
        axes = list(range(self.ndim(tensor)))
        if source < 0: source = axes[source]
        if target < 0: target = axes[target]
        try:
            axes.pop(source)
        except IndexError:
            raise ValueError('Source should verify 0 <= source < tensor.ndim'
                             'Got %d' % source)
        try:
            axes.insert(target, source)
        except IndexError:
            raise ValueError('Destination should verify 0 <= destination < tensor.ndim'
                             'Got %d' % target)
        return transpose(tensor, axes)

    @staticmethod
    def mean(tensor, axis=None, **kwargs):
        if axis is None:
            axis = ()
        res = nd.mean(tensor, axis=axis, **kwargs)
        if res.shape == (1,):
            return res.asscalar()
        else:
            return res

    @staticmethod
    def sum(tensor, axis=None, **kwargs):
        if axis is None:
            axis = ()
        res = nd.sum(tensor, axis=axis, **kwargs)
        if res.shape == (1,):
            return res.asscalar()
        else:
            return res

    @staticmethod
    def sqrt(tensor, *args, **kwargs):
        if isinstance(tensor, nd.NDArray):
            return nd.sqrt(tensor, *args, **kwargs)
        else:
            return math.sqrt(tensor)

    @staticmethod
    def copy(tensor):
        return tensor.copy()

    @staticmethod
    def concatenate(tensors, axis):
        return nd.concat(*tensors, dim=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return stack(*arrays, axis=axis)

    def symeig_svd(self, matrix, n_eigenvecs=None):
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
            U, S = nd.linalg.syevd(dot(matrix, transpose(matrix)))
            S = self.sqrt(S)
            V = dot(transpose(matrix), U / reshape(S, (1, -1)))
        else:
            V, S = nd.linalg.syevd(dot(transpose(matrix), matrix))
            S = self.sqrt(S)
            U = dot(matrix, V) / reshape(S, (1, -1))

        U, S, V = U[:, ::-1], S[::-1], transpose(V)[::-1, :]
        return U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'symeig_svd': self.symeig_svd}

for name in ['float64', 'float32', 'int64', 'int32']:
    MxnetBackend.register_method(name, getattr(numpy, name))

for name in ['arange', 'zeros', 'zeros_like', 'ones', 'eye', 'dot',
             'transpose', 'where', 'sign', 'prod', 'diag']:
    MxnetBackend.register_method(name, getattr(nd, name))
