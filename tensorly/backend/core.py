import inspect
import importlib
import os
import sys
import threading
import types
import warnings

import numpy as np
import scipy.linalg
import scipy.sparse.linalg


_STATE = threading.local()
_DEFAULT_BACKEND = None  # changed in `_initialize_default_backend`
_KNOWN_BACKENDS = ('numpy', 'mxnet', 'pytorch', 'tensorflow', 'cupy')
_LOADED_BACKENDS = {}


def _initialize_default_backend():
    """Setup the initial Tensorly backend"""
    default = 'numpy'
    backend = os.environ.get('TENSORLY_BACKEND', default)

    if backend not in _KNOWN_BACKENDS:
        msg = ("TENSORLY_BACKEND should be one of {%s}, got %r. Defaulting to "
               "'%r'") % (', '.join(map(repr, _KNOWN_BACKENDS)), backend, default)
        warnings.warn(msg, UserWarning)
        backend = default

    set_backend(backend, make_default=True)


class set_backend(object):
    """Sets the backend for TensorLy.

    Can be used to set the backend globally, or locally as a contextmanager.

    Parameters
    ----------
    backend : {'numpy', 'mxnet', 'pytorch', 'tensorflow', 'cupy'}
        The name of the backend to use. Default is 'numpy'.
    make_default : bool, optional
        If True, the backend will become the default backend for all threads.
        Note that this only affects threads where the backend hasn't already
        been explicitly set. If False (default) the backend is only set
        contextually for the local thread.

    Examples
    --------
    Set the backend to numpy globally for this thread:

    >>> import tensorly as tl
    >>> tl.set_backend('numpy')

    Set the backend to numpy globally for all threads:

    >>> tl.set_backend('numpy', make_default=True)

    Set the backend to use pytorch inside a context:

    >>> with tl.set_backend('pytorch'):  # doctest: +SKIP
    ...     pass
    """
    def __init__(self, backend, make_default=False):
        if not isinstance(backend, Backend):
            if backend not in _LOADED_BACKENDS:
                # load the backend
                if backend in _KNOWN_BACKENDS:
                    importlib.import_module('tensorly.backend.%s_backend' % backend)
                else:
                    msg = ("Unknown backend %r, known backends are {%s}"
                           % (backend, ', '.join(map(repr, _KNOWN_BACKENDS))))
                    raise ValueError(msg)
            backend = _LOADED_BACKENDS[backend]

        # Set the backend
        self._old_backend = getattr(_STATE, 'backend', None)
        self._new_backend = _STATE.backend = backend

        if make_default:
            global _DEFAULT_BACKEND
            _DEFAULT_BACKEND = backend

    def __repr__(self):
        return 'tensorly.set_backend(%r)' % self._new_backend.backend_name

    def __enter__(self):
        return None

    def __exit__(self, *args):
        if self._old_backend is None:
            del _STATE.backend
        else:
            _STATE.backend = self._old_backend


def get_backend():
    """Returns the backend currently used

    Returns
    -------
    backend_name : str
        The backend currently in use
    """
    return getattr(_STATE, 'backend', _DEFAULT_BACKEND).backend_name


class BackendAttributeModuleType(types.ModuleType):
    """A module type to dispatch backend generic attributes."""

    _dynamic_attributes = {'int64', 'int32', 'float64', 'float32', 'SVD_FUNS'}

    def __getattr__(self, key):
        if key in self._dynamic_attributes:
            backend = getattr(_STATE, 'backend', _DEFAULT_BACKEND)
            return getattr(backend, key)
        raise AttributeError("module %r has no attribute %r"
                             % (self.__name__, key))

    def __dir__(self):
        out = set(super(BackendAttributeModuleType, self).__dir__())
        out.update(self._dynamic_attributes)
        return list(out)


def wrap_module(module_name):
    """Wrap a module to dynamically dispatch attributes to the registry.

    Intended use is

    >>> registry.wrap_module(__name__)
    """
    sys.modules[module_name].__class__ = BackendAttributeModuleType


def register_backend(backend):
    """Register a backend with tensorly

    Parameters
    ----------
    backend : Backend
        The backend to register.
    """
    _LOADED_BACKENDS[backend.backend_name] = backend


class Backend(object):
    @classmethod
    def register_method(cls, name, func):
        """Register a method with the backend.

        Parameters
        ----------
        name : str
            The method name.
        func : callable
            The method
        """
        setattr(cls, name, staticmethod(func))

    @property
    def int64(self):
        raise NotImplementedError

    @property
    def int32(self):
        raise NotImplementedError

    @property
    def float64(self):
        raise NotImplementedError

    @property
    def float32(self):
        raise NotImplementedError

    @property
    def SVD_FUNS(self):
        raise NotImplementedError

    @staticmethod
    def context(tensor):
        """Returns the context of a tensor

        Creates a dictionary of the parameters characterising the tensor.

        Parameters
        ----------
        tensor : tensorly.tensor

        Returns
        -------
        context : dict

        Examples
        --------
        >>> import tensorly as tl
        >>> tl.set_backend('numpy')

        Imagine you have an existing tensor `tensor`:

        >>> tensor = tl.tensor([0, 1, 2], dtype=tl.float32)

        The context, here, will simply be the dtype:

        >>> tl.context(tensor)
        {'dtype': dtype('float32')}

        Note that, if you were using, say, PyTorch, the context would also
        include the device (i.e. CPU or GPU) and device ID.

        If you want to create a new tensor in the same context, use this context:

        >>> new_tensor = tl.tensor([1, 2, 3], **tl.context(tensor))
        """
        raise NotImplementedError

    @staticmethod
    def tensor(data, **context):
        """Tensor class

        Returns a tensor on the specified context, depending on the backend.

        Examples
        --------
        >>> import tensorly as tl
        >>> tl.set_backend('numpy')
        >>> tl.tensor([1, 2, 3], dtype=tl.int64)
        array([1, 2, 3])
        """
        raise NotImplementedError

    @staticmethod
    def is_tensor(obj):
        """Returns if `obj` is a tensor for the current backend"""
        raise NotImplementedError

    @staticmethod
    def shape(tensor):
        """Return the shape of a tensor"""
        raise NotImplementedError

    @staticmethod
    def ndim(tensor):
        """Return the number of dimensions of a tensor"""
        raise NotImplementedError

    @staticmethod
    def to_numpy(tensor):
        """Returns a copy of the tensor as a NumPy array.

        Parameters
        ----------
        tensor : tl.tensor

        Returns
        -------
        numpy_tensor : numpy.ndarray
        """
        raise NotImplementedError

    @staticmethod
    def copy(tensor):
        """Return a copy of the given tensor"""
        raise NotImplementedError

    @staticmethod
    def concatenate(tensors, axis=0):
        """Concatenate tensors along an axis.

        Parameters
        ----------
        tensors : list of tensor
            The tensors to concatenate. Non-empty tensors provided must have the
            same shape, except along the specified axis.
        axis : int, optional
            The axis to concatenate on. Default is 0.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def reshape(tensor, newshape):
        """Gives a new shape to a tensor without changing its data.

        Parameters
        ----------
        tensor : tl.tensor
        newshape : int or tuple of ints
            The new shape should be compatible with the original shape. If an
            integer, then the result will be a 1-D tensor of that length.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def transpose(tensor):
        """Permute the dimensions of a tensor.

        Parameters
        ----------
        tensor : tensor
        """
        raise NotImplementedError

    @staticmethod
    def moveaxis(tensor, source, destination):
        """Move axes of a tensor to new positions.

        Parameters
        ----------
        tensor : tl.tensor
        source : int or sequence of int
            Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
            Destination positions for each of the original axes. These must also be
            unique.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def arange(start=0, stop=None, step=None):
        """Return evenly spaced values within a given interval.

        Parameters
        ----------
        start : number, optional
            Start of the interval, inclusive. Default is 0.
        stop : number
            End of the interval, exclusive.
        step : number, optional
            Spacing between values. Default is 1.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def ones(shape, dtype=None):
        """Return a new tensor of given shape and type, filled with ones.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new tensor.
        dtype : data-type, optional
            The desired data-type for the tensor.
        """
        raise NotImplementedError

    @staticmethod
    def zeros(shape, dtype=None):
        """Return a new tensor of given shape and type, filled with zeros.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new tensor.
        dtype : data-type, optional
            The desired data-type for the tensor.
        """
        raise NotImplementedError

    @staticmethod
    def zeros_like(tensor):
        """Return at tensor of zeros with the same shape and type as a given tensor.

        Parameters
        ----------
        tensor : tensor
        """
        raise NotImplementedError

    @staticmethod
    def eye(N):
        """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        N : int
            Number of rows in the output.
        """
        raise NotImplementedError

    @staticmethod
    def where(condition, x, y):
        """Return elements, either from `x` or `y`, depending on `condition`.

        Parameters
        ----------
        condition : tensor
            When True, yield element from `x`, otherwise from `y`.
        x, y : tensor
            Values from which to choose.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        """Clip the values of a tensor to within an interval.

        Given an interval, values outside the interval are clipped to the interval
        edges.  For example, if an interval of ``[0, 1]`` is specified, values
        smaller than 0 become 0, and values larger than 1 become 1.

        Not more than one of `a_min` and `a_max` may be `None`.

        Parameters
        ----------
        tensor : tl.tensor
            The tensor.
        a_min : scalar, optional
            Minimum value. If `None`, clipping is not performed on lower bound.
        a_max : scalar, optional
            Maximum value. If `None`, clipping is not performed on upper bound.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def max(tensor):
        """The max value in a tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        scalar
        """
        raise NotImplementedError

    @staticmethod
    def min(tensor):
        """The min value in a tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        scalar
        """
        raise NotImplementedError

    @staticmethod
    def all(tensor):
        """Returns if all array elements in a tensor are True.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @staticmethod
    def mean(tensor, axis=None):
        """Compute the mean of a tensor, optionally along an axis.

        Parameters
        ----------
        tensor : tensor
        axis : int, optional
            If provided, the mean is computed along this axis.

        Returns
        -------
        out : scalar or tensor
        """
        raise NotImplementedError

    @staticmethod
    def sum(tensor, axis=None):
        """Compute the sum of a tensor, optionally along an axis.

        Parameters
        ----------
        tensor : tensor
        axis : int, optional
            If provided, the sum is computed along this axis.

        Returns
        -------
        out : scalar or tensor
        """
        raise NotImplementedError

    @staticmethod
    def prod(tensor, axis=None):
        """Compute the product of a tensor, optionally along an axis.

        Parameters
        ----------
        tensor : tensor
        axis : int, optional
            If provided, the product is computed along this axis.

        Returns
        -------
        out : scalar or tensor
        """
        raise NotImplementedError

    @staticmethod
    def sign(tensor):
        """Computes the element-wise sign of the given input tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        out : tensor
        """
        raise NotImplementedError

    @staticmethod
    def abs(tensor):
        """Computes the element-wise absolute value of the given input tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        out : tensor
        """
        raise NotImplementedError

    @staticmethod
    def sqrt(tensor):
        """Computes the element-wise sqrt of the given input tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        out : tensor
        """
        raise NotImplementedError

    @staticmethod
    def norm(tensor, order=2, axis=None):
        """Computes the l-`order` norm of a tensor.

        Parameters
        ----------
        tensor : tl.tensor
        order : int
        axis : int or tuple

        Returns
        -------
        float or tensor
            If `axis` is provided returns a tensor.
        """
        raise NotImplementedError

    @staticmethod
    def dot(a, b):
        """Dot product of two tensors.

        Parameters
        ----------
        a, b : tensor
            The tensors to compute the dot product of.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def solve(a, b):
        """Solve a linear matrix equation, or system of linear scalar equations.

        Computes the "exact" solution, `x`, of the well-determined, i.e., full
        rank, linear matrix equation `ax = b`.

        Parameters
        ----------
        a : tensor, shape (M, M)
            The coefficient matrix.
        b : tensor, shape (M,) or (M, K)
            The ordinate values.

        Returns
        -------
        x : tensor, shape (M,) or (M, K)
            Solution to the system a x = b. Returned shape is identical to `b`.
        """
        raise NotImplementedError

    @staticmethod
    def qr(a):
        """Compute the qr factorization of a matrix.

        Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
        upper-triangular.

        Parameters
        ----------
        a : tensor, shape (M, N)
            Matrix to be factored.

        Returns
        -------
        Q, R : tensor
        """
        raise NotImplementedError

    def kron(self, a, b):
        """Kronecker product of two tensors.

        Parameters
        ----------
        a, b : tensor
            The tensors to compute the kronecker product of.

        Returns
        -------
        tensor
        """
        s1, s2 = self.shape(a)
        s3, s4 = self.shape(b)
        a = self.reshape(a, (s1, 1, s2, 1))
        b = self.reshape(b, (1, s3, 1, s4))
        return self.reshape(a * b, (s1 * s3, s2 * s4))

    def kr(self, matrices):
        """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.

        Parameters
        ----------
        matrices : list of tensors
            List of 2D tensors with the same number of columns, i.e.::

                for i in len(matrices):
                    matrices[i].shape = (n_i, m)

        Returns
        -------
        khatri_rao_product : tensor of shape ``(prod(n_i), m)``
            Where ``prod(n_i) = prod([m.shape[0] for m in matrices])`` (i.e. the
            product of the number of rows of all the matrices in the product.)

        Notes
        -----
        Mathematically:

        .. math::
            \\text{If every matrix } U_k \\text{ is of size } (I_k \\times R),\\\\
            \\text{Then } \\left(U_1 \\bigodot \\cdots \\bigodot U_n \\right) \\\\
            text{ is of size } (\\prod_{k=1}^n I_k \\times R)
        """
        if len(matrices) < 2:
            raise ValueError('kr requires a list of at least 2 matrices, but {} '
                             'given.'.format(len(matrices)))

        n_col = self.shape(matrices[0])[1]
        for i, e in enumerate(matrices[1:]):
            if not i:
                res = matrices[0]
            s1, s2 = self.shape(res)
            s3, s4 = self.shape(e)
            if not s2 == s4 == n_col:
                raise ValueError('All matrices should have the same number of columns.')
            a = self.reshape(res, (s1, 1, s2))
            b = self.reshape(e, (1, s3, s4))
            res = self.reshape(a * b, (-1, n_col))
        return res

    def partial_svd(self, matrix, n_eigenvecs=None):
        """Computes a fast partial SVD on `matrix`

        If `n_eigenvecs` is specified, sparse eigendecomposition is used on
        either matrix.dot(matrix.T) or matrix.T.dot(matrix).

        Parameters
        ----------
        matrix : tensor
            A 2D tensor.
        n_eigenvecs : int, optional, default is None
            If specified, number of eigen[vectors-values] to return.

        Returns
        -------
        U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
            Contains the right singular vectors
        S : 1-D tensor, shape (n_eigenvecs, )
            Contains the singular values of `matrix`
        V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
            Contains the left singular vectors
        """
        # Check that matrix is... a matrix!
        if self.ndim(matrix) != 2:
            raise ValueError('matrix be a matrix. matrix.ndim is %d != 2'
                             % self.ndim(matrix))

        ctx = self.context(matrix)
        is_numpy = isinstance(matrix, np.ndarray)

        if not is_numpy:
            matrix = self.to_numpy(matrix)

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
        else:
            # We can perform a partial SVD
            # First choose whether to use X * X.T or X.T *X
            if dim_1 < dim_2:
                S, U = scipy.sparse.linalg.eigsh(
                    np.dot(matrix, matrix.T.conj()), k=n_eigenvecs, which='LM'
                )
                S = np.sqrt(S)
                V = np.dot(matrix.T.conj(), U * 1 / S[None, :])
            else:
                S, V = scipy.sparse.linalg.eigsh(
                    np.dot(matrix.T.conj(), matrix), k=n_eigenvecs, which='LM'
                )
                S = np.sqrt(S)
                U = np.dot(matrix, V) * 1 / S[None, :]

            # WARNING: here, V is still the transpose of what it should be
            U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
            V = V.T.conj()

        if not is_numpy:
            U = self.tensor(U, **ctx)
            S = self.tensor(S, **ctx)
            V = self.tensor(V, **ctx)
        return U, S, V


def dispatch(method):
    """Create a dispatched function from a generic backend method."""
    name = method.__name__

    def inner(*args, **kwargs):
        backend = getattr(_STATE, 'backend', _DEFAULT_BACKEND)
        return getattr(backend, name)(*args, **kwargs)

    # We don't use `functools.wraps` here because some of the dispatched
    # methods include the backend (`self`) as a parameter. Instead we manually
    # copy over the needed information, and filter the signature for `self`.
    for attr in ['__module__', '__name__', '__qualname__', '__doc__',
                 '__annotations__']:
        try:
            setattr(inner, attr, getattr(method, attr))
        except AttributeError:
            pass

    sig = inspect.signature(method)
    if 'self' in sig.parameters:
        parameters = [v for k, v in sig.parameters.items() if k != 'self']
        sig = sig.replace(parameters=parameters)
    inner.__signature__ = sig

    return inner


# Generic methods, exposed as part of the public API
context = dispatch(Backend.context)
tensor = dispatch(Backend.tensor)
is_tensor = dispatch(Backend.is_tensor)
shape = dispatch(Backend.shape)
ndim = dispatch(Backend.ndim)
to_numpy = dispatch(Backend.to_numpy)
copy = dispatch(Backend.copy)
concatenate = dispatch(Backend.concatenate)
reshape = dispatch(Backend.reshape)
transpose = dispatch(Backend.transpose)
moveaxis = dispatch(Backend.moveaxis)
arange = dispatch(Backend.arange)
ones = dispatch(Backend.ones)
zeros = dispatch(Backend.zeros)
zeros_like = dispatch(Backend.zeros_like)
eye = dispatch(Backend.eye)
where = dispatch(Backend.where)
clip = dispatch(Backend.clip)
max = dispatch(Backend.max)
min = dispatch(Backend.min)
all = dispatch(Backend.all)
mean = dispatch(Backend.mean)
sum = dispatch(Backend.sum)
prod = dispatch(Backend.prod)
sign = dispatch(Backend.sign)
abs = dispatch(Backend.abs)
sqrt = dispatch(Backend.sqrt)
norm = dispatch(Backend.norm)
dot = dispatch(Backend.dot)
kron = dispatch(Backend.kron)
solve = dispatch(Backend.solve)
qr = dispatch(Backend.qr)
kr = dispatch(Backend.kr)
partial_svd = dispatch(Backend.partial_svd)


_initialize_default_backend()
