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


class Index():
    """Convenience class used as a an array, to be used with index_update

    Parameters
    ----------
    indices : indices for indexing

    Examples
    --------
    Usage: index[indices], e.g. ::

        index[1:3, 4:5, :None]
    
    See also
    --------
    index_update : updating the values of a tensor for specified indices
    """
    __slots__ = ()

    def __getitem__(self, indices):
        return indices

    @property
    def __name__(self):
        return 'Index'

class Backend(object):
    _available_backends = dict()

    def __init_subclass__(cls, backend_name, **kwargs):
        """When a subclass is created, register it in _known_backends"""
        super().__init_subclass__(**kwargs)

        if backend_name != '':
            cls._available_backends[backend_name.lower()] = cls
            cls.backend_name = backend_name
        else:
            warnings.warn(f'Creating a subclass of BaseBackend ({cls.__name__}) with no name.')

    def __repr__(self):
        return f'TensorLy {self.backend_name}-backend'

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
    def complex128(self):
        raise NotImplementedError

    @property
    def complex64(self):
        raise NotImplementedError

    @staticmethod
    def check_random_state(seed):
        """Returns a valid RandomState

        Parameters
        ----------
        seed : None or instance of int or np.random.RandomState(), default is None
        if seed is None NumPy's global seed is used.

        Returns
        -------
        Valid instance np.random.RandomState

        Notes
        -----
        Inspired by the scikit-learn eponymous function
        """
        if seed is None:
            return np.random.mtrand._rand

        elif isinstance(seed, int):
            return np.random.RandomState(seed)

        elif isinstance(seed, np.random.RandomState):
            return seed

        raise ValueError('Seed should be None, int or np.random.RandomState')

    def randn(self, shape, seed=None, **context):
        """Returns a random tensor with samples from the “standard normal” distribution.

        Parameters
        ----------
        shape: Iterable[int]
            shape of the random tensor
        seed: None or instance of int or np.random.RandomState(), default is None
        if seed is None NumPy's global seed is used
        context: context of tensor

        Returns
        -------
        random_tensor: tl.tensor
        """
        rng = self.check_random_state(seed)
        random_tensor = rng.randn(*shape)
        random_tensor = self.tensor(random_tensor, **context)
        return random_tensor

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
    def diag(diagnoal):
        """Return a 2-D tensor with the elements of `diagonal` on the diagonal and zeros elsewhere.

        Parameters
        ----------
        diagonal : 1-D tensor
            diagonnal elements of the 2-D tensor to construct.
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
    def count_nonzero(tensor):
        """Returns number of non-zero elements in the tensor.
        Parameters
        ----------
        tensor : tensor
        Returns
        -------
        out : scalar
        """
        raise NotImplementedError

    @staticmethod
    def trace(tensor):
        """Returns sum of the elements on the diagonal of the tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        out : scalar or tensor
        """
        raise NotImplementedError

    @staticmethod
    def cumsum(tensor, axis=None):
        """Computes the cumulative sum of a tensor, optionally along an axis.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        tensor
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
    def any(tensor, axis=None, keepdims=False, **kwargs):
        """Test whether any array element along a given axis evaluates to True.

        Parameters
        ----------
        tensor : tensor
            input tensor to check for non-zero values
        axis : int or None, default is None
            optional, indicates an axis along which to check for non-zero values
        keepdims : bool, default is False

        Returns
        -------
        bool or tensor
            if axis is None, returns a bool indicating whether any value is non-zero
            otherwise, returns a tensor of bools.
        """
        return tensor.any(axis=axis, keepdims=keepdims, **kwargs)
          
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
    def argmax(tensor):
        """The argument of the max value in a tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        scalar
        """
        raise NotImplementedError

    @staticmethod
    def argmin(tensor):
        """The argument of the min value in a tensor.

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

    def norm(self, tensor, order=2, axis=None):
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
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == 'inf':
            return self.max(self.abs(tensor), axis=axis)
        if order == 1:
            return self.sum(self.abs(tensor), axis=axis)
        elif order == 2:
            return self.sqrt(self.sum(self.abs(tensor)**2, axis=axis))
        else:
            return self.sum(self.abs(tensor)**order, axis=axis)**(1 / order)

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
    def matmul(a, b):
        """Matrix multiplication of tensors representing (batches of) matrices

        Parameters
        ----------
        a : tl.tensor
            [description]
        b : tl.tensor
            tensors representing the matrices to contract

        Returns
        -------
        a @ b
            matrix product of a and b

        Notes
        -----
        The behavior depends on the arguments in the following way.

            * If both arguments are 2-D they are multiplied like conventional matrices.

            * If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

            * If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.

            * If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
        
        `matmul` differs from dot in two important ways:

           * Multiplication by scalars is not allowed, use * instead.

           * Stacks of matrices are broadcast together as if the matrices were elements, respecting the signature ``(n,k),(k,m)->(n,m)``:

           .. code-block:: python

              >>> a = np.ones([9, 5, 7, 4])

              >>> c = np.ones([9, 5, 4, 3])

              >>> np.dot(a, c).shape
              (9, 5, 7, 9, 5, 3)

              >>> np.matmul(a, c).shape
              (9, 5, 7, 3)

              >>> # n is 7, k is 4, m is 3

        The matmul function implements the semantics of the ``@`` operator introduced in Python 3.5 following `PEP 465 <https://www.python.org/dev/peps/pep-0465/>`_.
        """
        raise NotImplementedError


    @staticmethod
    def tensordot(a, b, axes=2):
        """
        Compute tensor dot product along specified axes.
        Given two tensors, `a` and `b`, and an array_like object containing
        two array_like objects, ``(a_axes, b_axes)``, sum the products of
        `a`'s and `b`'s elements (components) over the axes specified by
        ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
        integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
        of `a` and the first ``N`` dimensions of `b` are summed over.

        Parameters
        ----------
        a, b : array_like
            Tensors to "dot".
        axes : int or (2,) array_like
            * integer_like
            If an int N, sum over the last N axes of `a` and the first N axes
            of `b` in order. The sizes of the corresponding axes must match.
            * (2,) array_like
            Or, a list of axes to be summed over, first sequence applying to `a`,
            second to `b`. Both elements array_like must be of the same length.

        Returns
        -------
        output : ndarray
            The tensor dot product of the input.

        Notes
        -----
        Three common use cases are:
            * ``axes = 0`` : tensor product :math:`a\\otimes b`
            * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
            * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

        When `axes` is integer_like, the sequence for evaluation will be: first
        the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
        Nth axis in `b` last.
        When there is more than one axis to sum over - and they are not the last
        (first) axes of `a` (`b`) - the argument `axes` should consist of
        two sequences of the same length, with the first axis to sum over given
        first in both sequences, the second axis second, and so forth.
        The shape of the result consists of the non-contracted axes of the
        first tensor, followed by the non-contracted axes of the second.
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
    def lstsq(a, b):
        """Computes a solution to the least squares problem :math:`||ax-b||_F`

        If the coefficient martix is underdetermined (m<n) and multiple
        solutions exist, the min norm solution is returned.

        Parameters
        ----------
        a : tensor, shape (M, N)
            The coefficient matrix.
        b : tensor, shape (M,) or (M, K)
             The ordinate values.

        Returns
        -------
        x : tensor, shape (N,) or (N, K)
            Solution to the least squares problem :math:`||ax-b||_F`.
        residuals : tensor, shape (K,)
            Sums of squared residuals: Squared Euclidean 2-norm for each column in ax-b.
            If the rank of a is < N or M <= N, this is an empty tensor.
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

    @staticmethod
    def stack(arrays, axis=0):
        """
        Join a sequence of arrays along a new axis.
        """
        raise NotImplementedError

    def eps(self, dtype):
        """Returns the machine epsilon for a given floating point dtype

        Parameters
        ----------
        dtype : tensorly.dtype
            the dtype for which to get the machine epsilon
        
        Returns
        -------
        eps : machine epsilon for `dtype`
        """
        return self.finfo(dtype).eps

    def finfo(self, dtype):
        """Machine limits for floating point types.

        Parameters
        ----------
        dtype: float, dtype or instance
                Kind of floating point data-type about which to get information.
        """
        return np.finfo(self.to_numpy(self.tensor([], dtype=dtype)).dtype)

    @staticmethod
    def conj(x, *args, **kwargs):
        """Return the complex conjugate, element-wise.

            The complex conjugate of a complex number is obtained by 
            changing the sign of its imaginary part.
        """
        raise NotImplementedError

    @staticmethod
    def sort(tensor, axis, descending = False):
        """Return a sorted copy of an array

        Parameters
        ----------
        tensor : tensor
            An N-D tensor
        axis : int or None
            Axis along which to sort. If None, the array is flattened before sorting. The default is -1, which sorts along the last axis.
        descending : bool
            If True, values are sorted in descending order, otherwise in ascending.

        Returns
        -------
        sorted_tensor : tensor
            An N-D array, sorted copy of input tensor
        """
        raise NotImplementedError

    @staticmethod
    def argsort(tensor, axis,  descending = False):
        """Returns arguments of a sorted array

        Parameters
        ----------
        tensor : tensor
            An N-D tensor
        axis : int or None
            Axis along which to sort. If None, the array is flattened before sorting. The default is -1, which sorts along the last axis.
        descending : bool
            If True, values are sorted in descending order, otherwise in ascending.

        Returns
        -------
        list of scalar values
        """
        raise NotImplementedError

    @staticmethod
    def einsum(subscripts, *operands):
        """Evaluates the Einstein summation convention on the operands.

        Parameters
        ----------
        subscripts : str
            Specifies the subscripts for summation.

        *operands : list of tensors
            tensors for the operation

        Returns
        -------
        output : ndarray
            The calculation based on the Einstein summation convention

        Notes
        -----
        This is only available for certain backends.
        """
        raise NotImplementedError

    def moveaxis(self, tensor, source, destination):
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

        axes = list(range(self.ndim(tensor)))
        if source < 0: source = axes[source]
        if destination < 0: destination = axes[destination]
        try:
            axes.pop(source)
        except IndexError:
            raise ValueError('Source should verify 0 <= source < tensor.ndim'
                             'Got %d' % source)
        try:
            axes.insert(destination, source)
        except IndexError:
            raise ValueError('Destination should verify 0 <= destination < tensor.ndim'
                             'Got %d' % destination)
        return self.transpose(tensor, axes)
    
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

    def kr(self, matrices, weights=None, mask=None):
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
                if weights is None:
                    res = matrices[0]
                else:
                    res = matrices[0]*self.reshape(weights, (1, -1))
            s1, s2 = self.shape(res)
            s3, s4 = self.shape(e)
            if not s2 == s4 == n_col:
                raise ValueError('All matrices should have the same number of columns.')

            a = self.reshape(res, (s1, 1, s2))
            b = self.reshape(e, (1, s3, s4))
            res = self.reshape(a * b, (-1, n_col))

        m = self.reshape(mask, (-1, 1)) if mask is not None else 1

        return res*m

    def svd_flip(self, U, V, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        This function is borrowed from scikit-learn/utils/extmath.py
        Parameters
        ----------
        U : ndarray
            u and v are the output of `partial_svd`
        V : ndarray
            u and v are the output of `partial_svd`
        u_based_decision : boolean, (default=True)
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.
        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        if u_based_decision:
            # columns of U, rows of V
            max_abs_cols = self.argmax(self.abs(U), axis=0)
            signs = self.sign(
                self.tensor([U[i, j] for (i, j) in zip(max_abs_cols, range(self.shape(U)[1]))], **self.context(U))
            )
            U = U * signs
            if self.shape(V)[0] > self.shape(U)[1]:
                signs = self.concatenate((signs, self.ones(self.shape(V)[0] - self.shape(U)[1])))
            V = V * signs[:self.shape(V)[0]][:, None]
        else:
            # rows of V, columns of U
            max_abs_rows = self.argmax(self.abs(V), axis=1)
            signs = self.sign(
                self.tensor([V[i, j] for (i, j) in zip(range(self.shape(V)[0]), max_abs_rows)], **self.context(V))
            )
            V = V * signs[:, None]
            if self.shape(U)[1] > self.shape(V)[0]:
                signs = self.concatenate((signs, self.ones(self.shape(U)[1] - self.shape(V)[0])))
            U = U * signs[:self.shape(U)[1]]

        return U, V

    def svd(self, matrix):
        raise NotImplementedError
    
    def eigh(self, matrix):
        raise NotImplementedError

    def partial_svd(self, matrix, n_eigenvecs=None, flip=False, random_state=None, **kwargs):
        """Computes a fast partial SVD on `matrix`

        If `n_eigenvecs` is specified, sparse eigendecomposition is used on
        either matrix.dot(matrix.T) or matrix.T.dot(matrix).

        Parameters
        ----------
        matrix : tensor
            A 2D tensor.
        n_eigenvecs : int, optional, default is None
            If specified, number of eigen[vectors-values] to return.
        flip : bool, default is False
            If True, the SVD sign ambiguity is resolved by making the largest component
            in the columns of U, positive.
        random_state: {None, int, np.random.RandomState}
            If specified, use it for sampling starting vector in a partial SVD(scipy.sparse.linalg.eigsh)
        **kwargs : optional
            kwargs are used to absorb the difference of parameters among the other SVD functions

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
            warnings.warn('In partial_svd: converting to NumPy.'
                          ' Check SVD_FUNS for available alternatives if you want to avoid this.')

        # Choose what to do depending on the params
        dim_1, dim_2 = self.shape(matrix)
        min_dim, max_dim = (dim_1, dim_2) if dim_1 < dim_2 else (dim_2, dim_1)
        matrix = self.to_numpy(matrix)

        if (n_eigenvecs is None) or (n_eigenvecs >= min_dim):
            # Just perform trucated SVD
            full_matrices = (n_eigenvecs is None) or (n_eigenvecs > min_dim)
            # If n_eigenvecs == min_dim, we don't want full_matrices=True, it's super slow
            U, S, V = scipy.linalg.svd(matrix, full_matrices=full_matrices)
            U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        else:
            # We can perform a partial SVD
            rng = self.check_random_state(random_state)
            # initilize with [-1, 1] as in ARPACK
            v0 = rng.uniform(-1, 1, min_dim)

            # First choose whether to use X * X.T or X.T *X
            if dim_1 < dim_2:
                S, U = scipy.sparse.linalg.eigsh(
                    np.dot(matrix, matrix.T.conj()), k=n_eigenvecs, which='LM', v0=v0
                )
                S = np.sqrt(np.clip(S, 0, None))
                S = np.clip(S, np.finfo(S.dtype).eps, None)  # To avoid divide by zero warning on next line
                V = np.dot(matrix.T.conj(), U * np.where(np.abs(S) <= np.finfo(S.dtype).eps, 0, 1/S)[None, :])
                U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
                V, R = np.linalg.qr(V)
                V = V * (2*(np.diag(R) >= 0) - 1)  # we can't use np.sign because np.sign(0) == 0
            else:
                S, V = scipy.sparse.linalg.eigsh(
                    np.dot(matrix.T.conj(), matrix), k=n_eigenvecs, which='LM', v0=v0
                )
                S = np.sqrt(np.clip(S, 0, None))
                S = np.clip(S, np.finfo(S.dtype).eps, None)
                U = np.dot(matrix, V) * np.where(np.abs(S) <= np.finfo(S.dtype).eps, 0, 1/S)[None, :]
                U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
                U, R = np.linalg.qr(U)
                U = U * (2*(np.diag(R) >= 0) - 1)

            # WARNING: here, V is still the transpose of what it should be
            V = V.T.conj()

        if flip:
            U, V = self.svd_flip(U, V)

        if not is_numpy:
            U = self.tensor(U, **ctx)
            S = self.tensor(S, **ctx)
            V = self.tensor(V, **ctx)

        return U, S, V

    def truncated_svd(self, matrix, n_eigenvecs=None, **kwargs):
        """Computes a truncated SVD on `matrix` using the backends's standard SVD

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
        min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

        if n_eigenvecs is None:
            n_eigenvecs = max_dim

        if n_eigenvecs > max_dim:
            warnings.warn('Trying to compute SVD with n_eigenvecs={0}, which '
                          'is larger than max(matrix.shape)={1}. Setting '
                          'n_eigenvecs to {1}'.format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim

        full_matrices = n_eigenvecs > min_dim

        U, S, V = self.svd(matrix, full_matrices=full_matrices)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        return U, S, V

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
        min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

        if n_eigenvecs is None:
            n_eigenvecs = max_dim

        if n_eigenvecs > max_dim:
            warnings.warn('Trying to compute SVD with n_eigenvecs={0}, which '
                          'is larger than max(matrix.shape)={1}. Setting '
                          'n_eigenvecs to {1}'.format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim

        if dim_1 > dim_2:
            S, U = self.eigh(self.dot(matrix, self.transpose(matrix)))
            S = self.sqrt(self.clip(S, self.eps(S.dtype)))
            V = self.dot(self.transpose(matrix), U / self.reshape(S, (1, -1)))
        else:
            S, V = self.eigh(self.dot(self.transpose(matrix), matrix))
            S = self.sqrt(self.clip(S, self.eps(S.dtype)))
            U = self.dot(matrix, V) / self.reshape(S, (1, -1))

        U, S, V = self.flip(U, axis=1), self.flip(S), self.flip(self.transpose(V), axis=0)
        return U[:, :min(dim_1, n_eigenvecs)], S[:min(dim_1, dim_2, n_eigenvecs)], V[:min(dim_2, n_eigenvecs), :]

    def randomized_svd(self, matrix, n_eigenvecs=None, n_oversamples=5, n_iter=2, random_state=None, **kwargs):
        """Computes a truncated randomized SVD.

        If `n_eigenvecs` is specified, sparse eigendecomposition is used on
        either matrix.dot(matrix.T) or matrix.T.dot(matrix).

        Parameters
        ----------
        matrix : tensor
            A 2D tensor.
        n_eigenvecs : int, optional, default is None
            If specified, number of eigen[vectors-values] to return.
        n_oversamples: int, optional, default = 5
            rank overestimation value for finiding subspace with better allignment
        n_iter: int, optional, default = 2
            number of power iterations for the `randomized_range_finder` subroutine
        random_state: {None, int, np.random.RandomState}
        **kwargs : optional
            kwargs are used to absorb the difference of parameters among the other SVD functions

        Returns
        -------
        U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
            Contains the right singular vectors
        S : 1-D tensor, shape (n_eigenvecs, )
            Contains the singular values of `matrix`
        V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
            Contains the left singular vectors

        Notes
        -----
        This function is implemented based on Algorith 5.1 in `Finding structure with randomness:
        Probabilistic algorithms for constructing approximate matrix decompositions`
        - Halko et al (2009)
        """
        # Check that matrix is... a matrix!
        if self.ndim(matrix) != 2:
            raise ValueError('matrix be a matrix. matrix.ndim is %d != 2'
                             % self.ndim(matrix))

        dim_1, dim_2 = self.shape(matrix)
        min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

        if n_eigenvecs is None:
            n_eigenvecs = max_dim

        if n_eigenvecs > max_dim:
            warnings.warn('Trying to compute SVD with n_eigenvecs={0}, which '
                          'is larger than max(matrix.shape)={1}. Setting '
                          'n_eigenvecs to {1}'.format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim

        n_dims = min(n_eigenvecs + n_oversamples, max_dim)

        if dim_1 > dim_2 and n_eigenvecs > min(min_dim, n_dims) or \
           dim_1 < dim_2 and n_eigenvecs < min(min_dim, n_dims):
            # transpose matrix to keep the reduced matrix shape minimal
            matrix_T = self.transpose(matrix)
            Q = self.randomized_range_finder(matrix_T, n_dims=n_dims, n_iter=n_iter, random_state=random_state)
            Q_H = self.conj(self.transpose(Q))
            matrix_reduced = self.transpose(self.dot(Q_H, matrix_T))
            U, S, V = self.truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
            V = self.dot(V, self.transpose(Q))
        else:
            Q = self.randomized_range_finder(matrix, n_dims=n_dims, n_iter=n_iter, random_state=random_state)
            Q_H = self.conj(self.transpose(Q))
            matrix_reduced = self.dot(Q_H, matrix)
            U, S, V = self.truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
            U = self.dot(Q, U)

        return U, S, V

    def randomized_range_finder(self, A, n_dims, n_iter=2, random_state=None):
        """Computes an orthonormal matrix (Q) whose range approximates the range of A,  i.e., Q Q^H A ≈ A

        Parameters
        ----------
        A : 2D-array
        n_dims : int, dimension of the returned subspace
        n_iter : int, number of power iterations to conduct (default = 2)
        random_state: {None, int, np.random.RandomState}

        Returns
        -------
        Q : 2D-array
            of shape (A.shape[0], min(n_dims, A.shape[0], A.shape[1]))

        Notes
        -----
        This function is implemented based on Algorith 4.4 in `Finding structure with randomness:
        Probabilistic algorithms for constructing approximate matrix decompositions`
        - Halko et al (2009)
        """
        rng = self.check_random_state(random_state)
        dim_1, dim_2 = self.shape(A)
        Q = self.tensor(rng.normal(size=(dim_2, n_dims)), **self.context(A))
        Q, _ = self.qr(self.dot(A, Q))

        # Perform power iterations when spectrum decays slowly
        A_H = self.conj(self.transpose(A))
        for i in range(n_iter):
            Q, _ = self.qr(self.dot(A_H, Q))
            Q, _ = self.qr(self.dot(A, Q))

        return Q

    index = Index()

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'truncated_svd': self.truncated_svd,
                'symeig_svd': self.symeig_svd,
                'randomized_svd': self.randomized_svd}
    
    @staticmethod
    def index_update(tensor, indices, values):
        """Updates the value of tensors in the specified indices
            Should be used as::

                index_update(tensor, tensorly.index[:, 3:5], values)

            Equivalent of::
            
                tensor[:, 3:5] = values

        Parameters
        ----------
        tensor : tensorly.tensor
            intput tensor which values to update
        indices : tensorly.index
            indices to update
        values : tensorly.tensor
            values to use to fill tensor[indices]
        
        Returns
        -------
        tensor
            updated tensor

        Example
        -------
        
        >>> import tensorly as tl
        >>> import numpy as np
        >>> tensor = tl.tensor([[1, 2, 3], [4, 5, 6]])
        >>> cpy = tensor.copy()
        >>> tensor[:, 1] = 0
        >>> tensor
        array([[1, 0, 3],
                [4, 0, 6]])
        >>> tl.index_update(tensor, tl.index[:, 1], 0)
        array([[1, 0, 3],
               [4, 0, 6]])

        See also
        --------
        index
        """
        tensor[indices] = values
        return tensor

    @staticmethod
    def log2(x):
        """Return the base 2 logarithm of x.
        """
        raise NotImplementedError

    @staticmethod
    def sin(x):
        """Return the sin of x.
        """
        raise NotImplementedError

    @staticmethod
    def cos(x):
        """Return the cos of x.
        """
        raise NotImplementedError

    @staticmethod
    def flip(tensor, axis=None):
        """Reverse the order of elements in an array along the given axis.
        """
        raise NotImplementedError