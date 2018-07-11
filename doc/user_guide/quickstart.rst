Quick-Start
===========

A short overview of TensorLy to get started quickly.

Tensor operations
-----------------

First import TensorLy:

.. code-block:: python

   import tensorly as tl

In the code written in TensorLy, you may notice we use function from tensorly rather than, say, NumPy. This is because we support several backends and we want the correct function to be called depending on the backend. 
For instance `tensorly.max` calls either the MXNet, NumPy or PyTorch version depending on the backend. There are other subtlties that are handled by the backend to allow a common API regardless of the backend use.

.. note::
   
   By default, the backend is set to NumPy. You can change the backend using `tensorly.set_backend`.
   For more information on the backend, refer to :doc:`./backend`.

Tensors can be created, e.g. from numpy arrays:

.. code-block:: python

   import numpy as np

   # create a random 10x10x10 tensor
   tensor = np.random.random((10, 10, 10))

You can then easily perform basic tensor operations:

.. code-block:: python
   
   # mode-1 unfolding (i.e. zeroth mode)
   unfolded = tl.unfold(tensor, mode=0)
   # refold the unfolded tensor
   tl.fold(unfolded, mode=0, shape=tensor.shape)
   
Tensor algebra
--------------

More '*advanced*' tensor algebra functions are located in the aptly named :py:mod:`tensorly.tenalg` module.

Tensor decomposition
--------------------

Decompositions are in the :py:mod:`tensorly.decomposition` module.

.. code-block:: python

   from tensorly.decomposition import tucker, parafac, non_negative_tucker

   # decompositions are one-liners:
   factors = parafac(tensor, rank=5)
   core, factors = tucker(tensor, ranks=[5, 5, 5])
   core, factors = non_negative_tucker(tensor, ranks=[5, 5, 5])

Tensor regressions
------------------

Located in the :py:mod:`tensorly.regression` module, tensor regression are objects that have a scikit-learn-like API, with a fit method for optimising the parameters and a predict one for applyting the method to new unseen data.

Metrics
-------

Whether you are training a tensor regression method or combining deep learning and tensor methods, you will need metrics to train and assess your method. These are implemented in the :py:mod:`tensorly.metrics` module

Sampling random tensors
-----------------------

To create random tensors, you can use the :py:mod:`tensorly.random` module.
