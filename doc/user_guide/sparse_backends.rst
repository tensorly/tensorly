.. _sparse_backend:

Sparse Backend
==============

TensorLy supports sparse tensors for some backends and algorithms.

When selecting a backend to be used for tensorly, this backend will also
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

Algorithms
~~~~~~~~~~

- ``parafac``
- ``tucker``
- ``robust_pca``

Usage
=====

The sparse sub-package in ``tensorly.contrib.sparse`` contains a mirror of the
interfaces in the usual, dense ``tensorly`` package. For example, unfolding a
sparse tensor would use::

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

Note that not all algorithms currently support sparse. This is because some
algorithms may need to be rewritten to properly avoid densifying large sparse
arrays, or creating arrays with large shapes. The algorithms listed above have
been tested with sparse tensors. Other algorithms may work, but one should be
careful to watch tensorly's memory usage when using them. With the
PyData/Sparse backend, you can set the environment variable
``SPARSE_AUTO_DENSIFY=0`` before importing ``sparse`` and it will cause it to
raise a ``RuntimeError`` whenever a sparse tensor would be automatically
densified. However, be aware that this does not protect against an algorithm
attempting to create a new dense array with a large shape.

Another caveat to be aware of is that some algorithms may perform better with
the dense variant, even when using sparse arrays. For example, the ``parafac``
algorithm has been written so as to avoid large memory usage for dense arrays.
However, because the resulting decomposed vectors will generally be dense even
if the input tensor is sparse. Thus, using the sparse ``parafac``
(``tensorly.contrib.sparse.decomposition.parafac``) may be slower and use more
memory than the dense version (``tensorly.decomposition.parafac``) for a
sparse input tensor.
