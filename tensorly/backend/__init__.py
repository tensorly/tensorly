from ..core import Registry as _Registry

_generics = _Registry()
_generics.wrap_module(__name__)

for attr in ['int32', 'int64', 'float32', 'float64']:
    _generics.add_attribute(attr)

_generics.add_attribute('SVD_FUNS')


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
def is_tensor(obj):
    """Returns if `obj` is a tensor for the current backend"""
    pass


@_generics.add_method
def to_numpy(tensor):
    """Returns a copy of the tensor as a NumPy array.

    Parameters
    ----------
    tensor : tl.tensor

    Returns
    -------
    numpy_tensor : numpy.ndarray
    """
    pass


@_generics.add_method
def shape(tensor):
    """Return the shape of a tensor"""
    pass


@_generics.add_method
def ndim(tensor):
    """Return the number of dimensions of a tensor"""
    pass


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
def copy(tensor):
    """Return a copy of the given tensor"""
    pass


@_generics.add_method
def transpose(tensor):
    """Permute the dimensions of a tensor.

    Parameters
    ----------
    tensor : tensor
    """
    pass


@_generics.add_method
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
    pass


@_generics.add_method
def ones(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor.
    dtype : data-type, optional
        The desired data-type for the tensor.
    """
    pass


@_generics.add_method
def zeros(shape, dtype=None):
    """Return a new tensor of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor.
    dtype : data-type, optional
        The desired data-type for the tensor.
    """
    pass


@_generics.add_method
def zeros_like(tensor):
    """Return at tensor of zeros with the same shape and type as a given tensor.

    Parameters
    ----------
    tensor : tensor
    """
    pass


@_generics.add_method
def eye(N):
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    """
    pass


@_generics.add_method
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
    pass


@_generics.add_method
def kron(a, b):
    """Kronecker product of two tensors.

    Parameters
    ----------
    a, b : tensor
        The tensors to compute the kronecker product of.

    Returns
    -------
    tensor
    """
    pass


@_generics.add_method
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
    pass


@_generics.add_method
def max(tensor):
    """The max value in a tensor.

    Parameters
    ----------
    tensor : tensor

    Returns
    -------
    scalar
    """
    pass


@_generics.add_method
def min(tensor):
    """The min value in a tensor.

    Parameters
    ----------
    tensor : tensor

    Returns
    -------
    scalar
    """
    pass


@_generics.add_method
def all(tensor):
    """Returns if all array elements in a tensor are True.

    Parameters
    ----------
    tensor : tensor

    Returns
    -------
    bool
    """
    pass


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
def sign(tensor):
    """Computes the element-wise sign of the given input tensor.

    Parameters
    ----------
    tensor : tensor

    Returns
    -------
    out : tensor
    """
    pass


@_generics.add_method
def abs(tensor):
    """Computes the element-wise absolute value of the given input tensor.

    Parameters
    ----------
    tensor : tensor

    Returns
    -------
    out : tensor
    """
    pass


@_generics.add_method
def sqrt(tensor):
    """Computes the element-wise sqrt of the given input tensor.

    Parameters
    ----------
    tensor : tensor

    Returns
    -------
    out : tensor
    """
    pass


@_generics.add_method
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
    pass


@_generics.add_method
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
    pass


@_generics.add_method
def kr(matrices):
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
    pass


@_generics.add_method
def partial_svd(matrix, n_eigenvecs=None):
    """Computes a fast partial SVD on `matrix`

    If `n_eigenvecs` is specified, sparse eigendecomposition is used on either
    matrix.dot(matrix.T) or matrix.T.dot(matrix).

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
    pass


@_generics.add_method
def assert_array_equal(a, b, **kwargs):
    pass


@_generics.add_method
def assert_array_almost_equal(a, b, **kwargs):
    pass


@_generics.add_method
def assert_raises(*args, **kwargs):
    pass


@_generics.add_method
def assert_equal(*args, **kwargs):
    pass


@_generics.add_method
def assert_(*args, **kwargs):
    pass
