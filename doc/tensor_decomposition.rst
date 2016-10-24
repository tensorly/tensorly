Tensor decomposition
====================

In this tutorial we will go over how to perform tensor decomposition.
Refer to [1]_ for more information on tensor decomposition.

CANDECOMP-PARAFAC
-----------------

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


References
----------
.. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
