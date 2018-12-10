.. _sparse_backend:

Sparse Backend Development
==========================

TensorLy's sparse support exists in the ``tensorly.contrib.sparse`` sub-package.
This is to enable fast iteration of sparse support.  The sparse sub-package
contains a mirror of the interfaces in the usual, dense ``tensorly`` package.
For example, unfolding a sparse tensor would use::

    >>> import numpy as np
    >>> import sparse
    >>> from tensorly.contrib.sparse import tensor, unfold
    >>> X = sparse.COO(np.arange(24).reshape((3, 4, 2)))
    >>> X = sparse.COO(X.coords, X.data, shape=(100, 100, 100))
    >>> X = tl.tensor(X)
    >>> unfold(X, 0) # mode-1 unfolding
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23]])

This separation makes it explicit whether a sparse or dense algorithm is
being used.

Backends
--------
When users select a backend to be used for tensorly, this backend will also
be applied to the sparse implementations. This is because many backends
naively support both dense and sparse tensors as distinct objects. For example,
TensorFlow and PyTorch both support separate interfaces for dense and sparse
representations. Using TensorLy's dense or sparse interfaces will give you
appropriate objects for the selected backend.

However, not all backends have a native sparse representation. For instance,
NumPy lacks a sparse object. In these cases, TensorLy makes opinionated decisions
about how to handle sparse support, if at all. It is usually necessary for
non-native sparse support to require additional dependencies. For the NumPy
backend, the `PyData/Sparse <https://sparse.pydata.org>`_ project is used
as the sparse representation.
