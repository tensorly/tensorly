Tucker form of a tensor
=======================

The Tucker decomposition can be seen as a generalisation of the CP decomposition: it decomposes the tensor into a small core tensor and factor matrices. CP can be seen as a Tucker decomposition with a super-diagonal core.

A tensor in its decomposed Tucker form is therefore nothing more than a core tensor with the same order as the original tensor and a list of projection matrices, one for each mode of the core tensor.

Tucker decomposition
---------------------


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
   [[  0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00   0.000e+00]
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


