Kruskal form of a tensor
========================

The idea is to express the tensor as a sum of rank one tensors. That is, a sum of outer product of vectors.
Such representation can be obtained by applying Canonical Polyadic Decomposition (also known as CANDECOMP-PARAFAC, CP, or PARAFAC decomposition). 

CANDECOMP-PARAFAC decomposition
-------------------------------

We demonstrate here how to perform a Canonical Polyadic Decomposition. A rank-r Parafac decomposes a tensor into a linear combination of r rank-1 tensors (See [1]_ for more details).

First, let's create a second order tensor that is zero everywhere except in a swiss shape that is one.

.. code-block::python
   >>> import numpy as np
   >>> import tensorly as tl
   >>> tensor = tl.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
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

