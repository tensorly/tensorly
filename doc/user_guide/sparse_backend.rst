.. _sparse_backend:

Sparse Backend
==============

TensorLy supports sparse tensors for some backends and algorithms.

When selecting a backend to be used for tensorly, this backend will also
be applied to the sparse implementations. This is because many backends
natively support both dense and sparse tensors as distinct objects. For example,
TensorFlow and PyTorch both have (some) support for sparse tensors.
Using TensorLy's dense or sparse interfaces will give you
appropriate objects for the selected backend.

Why a separate sparse backend?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some algorithms need to be adapted to be *sparse-safe*. 

In addition, not all backends have a native sparse representation. For instance,
NumPy lacks a sparse array structure.
In these cases, TensorLy makes opinionated decisions
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
-----

The sparse sub-package in ``tensorly.contrib.sparse`` contains a mirror of the
interfaces in the usual, dense ``tensorly`` package. For example, unfolding a
sparse tensor would use::

    >>> import numpy as np
    >>> import sparse
    >>> from tensorly.contrib.sparse import tensor, unfold
    >>> X = sparse.COO(np.arange(24).reshape((3, 4, 2)))
    >>> X = sparse.COO(X.coords, X.data, shape=(100, 100, 100))
    >>> X = tensor(X, dtype='float')
    >>> unfold(X, 0).data # mode-1 unfolding
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23]])

This separation makes it explicit whether a sparse or dense algorithm is
being used.

Note that not all algorithms currently support sparse. This is because some
algorithms may need to be rewritten to properly avoid densifying large sparse
arrays, or creating arrays with large shapes. The algorithms listed above have
been tested with sparse tensors. Other algorithms may work, but one should be
careful to watch tensorly's memory usage when using them. 

With the PyData/Sparse backend, you can set the environment variable
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

Missing Values
~~~~~~~~~~~~~~

The ``parafac`` function supports masks for missing values. A mask should be a
boolean array of the same shape as the original tensor that is ``False``/``0``
where the value is missing and ``True``/``1`` where it is not. It is passed to
the ``parafac()`` function via the ``mask`` parameter.

When using masks with sparse tensors, there are two important caveats:

- The mask itself should be sparse. In the algorithm, the memory used will be
  proportional to the number of missing values. Values of the original sparse
  tensor that are ``0`` should be ``True``/``1`` (i.e., non-missing) in the
  mask.

- Sparse mask support is currently only supported with the sparse backend
  ``parafac`` (``tensor.contrib.sparse.decomposition.parafac``). The dense
  backend ``parafac`` will densify the array.

Example
-------

In the following example, we construct a random sparse tensor that has a known
rank (by construction), and decompose it with parafac.

.. code:: python

   >>> import tensorly.contrib.sparse as stl; import sparse
   >>> shape = (1000, 1001, 1002)
   >>> rank = 5
   >>> starting_weights = stl.ones((rank))
   >>> starting_weights
   <COO: shape=(5,), dtype=float64, nnz=0, fill_value=1.0>
   >>> starting_factors = [sparse.random((i, rank)) for i in shape]
   >>> starting_factors
   [<COO: shape=(1000, 5), dtype=float64, nnz=50, fill_value=0.0>, <COO: shape=(1001, 5), dtype=float64, nnz=50, fill_value=0.0>, <COO: shape=(1002, 5), dtype=float64, nnz=50, fill_value=0.0>]

Here we construct a tensor from the random factors. Note that in general,
a recomposed tensor will be dense, but for our constructed example it is
sparse, so we can use ``cp_to_tensor`` without worrying about using too
much memory.

   >>> from tensorly.contrib.sparse.cp_tensor import cp_to_tensor
   >>> tensor = cp_to_tensor((starting_weights, starting_factors))
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
   >>> t = time.time(); dense_cp = parafac(tensor, 5, init='random'); print(time.time() - t)
   1.3858051300048828

Note that the decomposition takes much longer when using the sparse variant.

   >>> from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac # The sparse version
   >>> t = time.time(); sparse_cp = sparse_parafac(tensor, 5, init='random'); print(time.time() - t)
   14.053689002990723

However, there can be advantages to using the sparse variant. It is currently
required when using ``init='svd'`` to make TensorLy use the sparse SVD
algorithm (from ``scipy.sparse``). Choosing the sparse backend ``parafac``
also makes it use ``scipy.sparse.linalg.spsolve`` instead of
``numpy.linalg.solve``, which can have advantages, for instance, when using
the scikit-umfpack backend, it is more robust against nearly singular
intermediate matrices.
