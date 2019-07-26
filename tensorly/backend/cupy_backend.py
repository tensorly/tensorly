try:
    import cupy as cp
except ImportError as error:
    message = ('Impossible to import cupy.\n'
               'To use TensorLy with the cupy backend, '
               'you must first install cupy!')
    raise ImportError(message) from error

import warnings
import numpy as np

from .core import Backend


class CupyBackend(Backend):

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=cp.float32):
        return cp.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, cp.ndarray)

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, cp.ndarray):
            return cp.asnumpy(tensor)
        return tensor

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return cp.clip(tensor, a_min, a_max)

    def norm(self, tensor, order=2, axis=None):
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
            return self.to_numpy(res)
        return res

    def solve(self, matrix1, matrix2):
        try:
            cp.linalg.solve(matrix1, matrix2)
        except cp.cuda.cusolver.CUSOLVERError:
            warnings.warn('CuPy solver failed, using numpy.linalg.solve instead.')
            ctx = self.context(matrix1)
            matrix1 = self.to_numpy(matrix1)
            matrix2 = self.to_numpy(matrix2)
            res = np.linalg.solve(matrix1, matrix2)
            return self.tensor(res, **ctx)

    @staticmethod
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

        U, S, V = cp.linalg.svd(matrix, full_matrices=full_matrices)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        return U, S, V

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'truncated_svd': self.truncated_svd}


for name in ['float64', 'float32', 'int64', 'int32', 'reshape', 'moveaxis',
             'transpose', 'copy', 'ones', 'zeros', 'zeros_like', 'eye',
             'arange', 'where', 'dot', 'kron', 'qr', 'concatenate', 'max',
             'min', 'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'stack',
             'conj', 'diag']:
    CupyBackend.register_method(name, getattr(cp, name))

