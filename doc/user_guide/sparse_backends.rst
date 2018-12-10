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
- ``non_negative_parafac``
- ``partial_tucker``
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

Example
=======

In the following example, we construct a random sparse tensor that has a known
rank (by construction), and decompose it with parafac.

.. code:: python

   >>> shape = (1000, 1001, 1002)
   >>> rank = 5
   >>> starting_factors = [sparse.random((i, rank)) for i in shape]
   >>> starting_factors
   [<COO: shape=(1000, 5), dtype=float64, nnz=50, fill_value=0.0>, <COO: shape=(1001, 5), dtype=float64, nnz=50, fill_value=0.0>, <COO: shape=(1002, 5), dtype=float64, nnz=50, fill_value=0.0>]

Here we construct a tensor from the random factors. Note that in general,
a recomposed tensor will be dense, but for our constructed example it is
sparse, so we can use ``kruskal_to_tensor`` without worrying about using too
much memory.

   >>> from tensorly.contrib.sparse.kruskal_tensor import kruskal_to_tensor
   >>> tensor = kruskal_to_tensor(starting_factors)
   >>> tensor
   <COO: shape=(1000, 1001, 1002), dtype=float64, nnz=5044, fill_value=0.0>

This is how much memory the sparse array takes up, vs. how much it would take
up if it were represented densly.

   >>> tensor.nbytes / 1e9 # Actual memory usage in GB
   0.000161408
   >>> import numpy as np
   >>> np.prod(tensor.shape) * 8 / 1e9 # Memory usage if array was dense, in GB
   8.024016

Now to decompose the tensor.

   >>> from tensorly.decomposition import parafac # The dense version
   >>> import time
   >>> t = time.time(); factors = parafac(tensor, 5, init='random'); print(time.time() - t)
   1.3858051300048828

Note that the decomposition takes much longer when using the sparse variant.

   >>> from tensorly.contrib.sparse.decomposition import parafac as parafac_sparse # The sparse version
   >>> t = time.time(); factors_sparse = parafac_sparse(tensor, 5, init='random'); print(time.time() - t)
   14.053689002990723

However, there can be advantages to using the sparse variant. It is currently
required when using ``init='svd'`` to make TensorLy use the sparse SVD
algorithm (from ``scipy.sparse``). Choosing the sparse backend ``parafac``
also makes it use ``scipy.sparse.linalg.spsolve`` instead of
``numpy.linalg.solve``, which can have advantages, for instance, when using
the scikit-umfpack backend, it is more robust against nearly singular
intermediate matrices.

For more examples of Decompositions with sparse backends, see the example
notebooks at
https://github.com/JeanKossaifi/tensorly-notebooks/tree/master/07_pydata_sparse_backend.
