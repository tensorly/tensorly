Tensor decomposition
====================

In this tutorial we will go over how to perform tensor decomposition.
Refer to [1]_ for more information on tensor decomposition.

CANDECOMP-PARAFAC
-----------------
We demonstrate here how to perform a Canonical Polyadic Decomposition (also known as CANDECOMP-PARAFAC, CP, or PARAFAC decomposition). A rank-r Parafac decomposes a tensor into a linear combination of r rank-1 tensors (See [1]_ for more details).

First, let's create a second order tensor that is zero everywhere except in a swiss shape that is one.

.. code-block::python
   >>> import numpy as np
   >>> import tensorly as tl
   >>> tensor = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                          [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                          [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                          [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                          [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])


We will now apply a rank-2 CANDECOMP-PARAFAC (:func:`tensorly.decomposition.parafac`) decomposition on `tensor`
to decompose this into a kruskal tensor.

A Parafac decompositions expresses the tensor as a kruskal tensor that can be represented as a list of factors (matrices).
The :func:`parafac` function therefore returns a list of factors.

.. code::

   >>> from tensorly.decomposition import parafac
   >>> factors = parafac(tensor, rank=2)
   >>> len(factors)
   2
   >>> [f.shape for f in factors]
   [(12, 2), (12, 2)]

From this **kruskal tensor** (presented as a list of matrices) you can reconstruct a full tensor:

.. code::

   >>> print(tl.kruskal_to_tensor(factors))
   [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.]
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.]
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.]
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.]
    [ 0.  0.  0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  1.  1.  1.  1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]


Tucker
------
The Tucker decomposition can be seen as a generalisation of the CP decomposition: it decomposes the tensor into a small core tensor and factor matrices. CP can be seen as a Tucker decomposition with a super-diagonal core.

Tucker (classical and non-negative) are available in TensorLy (:func:`tensorly.decomposition.tucker` and :func:`tensorly.decomposition.non_negative_tucker`).

Using the same tensor as previously, we will perform a rank [2, 3]-decomposition of `tensor`:

.. code::

   >>> from tensorly.decomposition import tucker
   >>> core, factors = tucker(tensor, ranks=[2, 3])
   # The core is a smaller tensor of size (2, 3):
   >>> core.shape
   (2, 3)
   >>> len(factors)
   2
   >>> [f.shape for f in factors]
   [(12, 2), (12, 3)]

As previously, we can reconstruct a full tensor from our Tucker decomposition:

.. code:: python
   
   >>> from tensorly import tucker_to_tensor
   >>> print(tucker_to_tensor(core, factors)
   [  0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00]
    [ -7.340e-17   2.617e-16   1.914e-16   2.475e-16   1.000e+00   1.000e+00   1.000e+00   1.000e+00   2.475e-16   2.475e-16   2.475e-16   0.000e+00]
    [ -7.340e-17   2.617e-16   1.914e-16   2.475e-16   1.000e+00   1.000e+00   1.000e+00   1.000e+00   2.475e-16   2.475e-16   2.475e-16   0.000e+00]
    [ -7.340e-17   2.617e-16   1.914e-16   2.475e-16   1.000e+00   1.000e+00   1.000e+00   1.000e+00   2.475e-16   2.475e-16   2.475e-16   0.000e+00]
    [  7.746e-17   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   0.000e+00]
    [  7.746e-17   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   0.000e+00]
    [  7.746e-17   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   0.000e+00]
    [  7.746e-17   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   1.000e+00   0.000e+00]
    [ -7.340e-17   2.617e-16   1.914e-16   2.475e-16   1.000e+00   1.000e+00   1.000e+00   1.000e+00   2.475e-16   2.475e-16   2.475e-16   0.000e+00]
    [ -7.340e-17   2.617e-16   1.914e-16   2.475e-16   1.000e+00   1.000e+00   1.000e+00   1.000e+00   2.475e-16   2.475e-16   2.475e-16   0.000e+00]
    [ -7.340e-17   2.617e-16   1.914e-16   2.475e-16   1.000e+00   1.000e+00   1.000e+00   1.000e+00   2.475e-16   2.475e-16   2.475e-16   0.000e+00]
    [  0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00]]

Note that some coefficients are almost zero (10e-16) but not exactly due to numerical approximations.

References
----------
.. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.

