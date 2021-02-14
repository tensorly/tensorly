Quick-Start
===========

A short overview of TensorLy to get started quickly and get familiar with the organization of TensorLY. 

Organization of TensorLy
-------------------------

TensorLy is organized in several submodule:

================================= ================================
Module                             Description
================================= ================================
:mod:`tensorly`                   Core operations,  includes regular array manipulation and tensor specific ones. 
:mod:`tensorly.tucker_tensor`     Manipulate tensors in decomposed Tucker form
:mod:`tensorly.cp_tensor`         Manipulate tensors decomposed in CP (or Parafac) form
:mod:`tensorly.tt_tensor`         Manipulate tensors in decomposed Tensor-Train format
:mod:`tensorly.tt_matrix`         Manipulate tensors decomposed in the TT-Matrix format
:mod:`tensorly.parafac2_tensor`   Manipulate tensors in decomposed PARAFAC-2 form
:mod:`tensorly.decomposition`     Perform tensor decomposition
:mod:`tensorly.regression`        Perform (low-rank) tensor regression
:mod:`tensorly.random`            Sample random tensors
:mod:`tensorly.metrics`           Error measures
:mod:`tensorly.contrib`           Experimental features, including sparse tensor decomposition and cross decomposition
:mod:`tensorly.datasets`          Loading data
================================= ================================


TensorLy Backend
----------------
Earlier, we mentioned that all function for manipulating arrays can be accessed through :mod:`tensorly` or `tensorly.backend`. 

For instance, if you have a tensor ``t``, to take its mean, you should use ``tensorly.mean(t)``, **not**, for instance, ``numpy.mean(t)`` (or torch, mxnet, etc).
Why is that? 

.. important:: 

   This is because we support several backends: the code you write in TensorLy can be *transparently* executed with several frameworks, without having to change anything in your code!
   For instance, you can execute your code normally using NumPy, but you can also have it run on GPU or multiple machines, using PyTorch, TensorFlow, CuPy, MXNet or JAX. Without having to adapt your code!


This is why you should always manipulate tensors using tensorly backend functions only.
For instance, `tensorly.max` calls either the MXNet, NumPy or PyTorch version depending on the backend you selected. There are other subtlties that are handled by the backend to allow a common API regardless of the backend use.

.. note::
   
   By default, the backend is set to NumPy. You can change the backend using ``tensorly.set_backend``.
   For instance, to switch to pytorch, simply type ``tensorly.set_backend('pytorch')``.
   For more information on the backend, refer to :doc:`./backend`.


Tensors can be created, e.g. from numpy arrays:

.. code-block:: python

   import tensorly as tl
   from tensorly import random

Now, let's create a random tensor of size 10x10x10:

.. code-block:: python

   tensor = random.random_tensor((10, 10, 10))
   # This will be a NumPy array by default

Now, if you want to use PyTorch instead:

.. code-block:: python

   tl.set_backend('pytorch')
   # TensorLy now uses TensorLy for all operations

   tensor = random.random_tensor((10, 10, 10))
   # This will be a PyTorch array by default


In all cases, you manipulate tensors in the same way:

.. code-block:: python

   tl.max(tensor)
   tl.mean(tensor)
   tl.dot(tl.unfold(tensor, 0), tl.transpose(tl.unfold(tensor, 0)))

Note that you can also access the backend functions explicitely through `tensorly.backend`:

.. code-block:: python

   import tensorly.backend as T

   T.max(tensor)



Tensor manipulation
-------------------

You can then easily perform basic :mod:`tensor operations <tensorly.base>`, such as folding, unfolding, etc:

.. code-block:: python
   
   # mode-1 unfolding (i.e. zeroth mode)
   unfolded = tl.unfold(tensor, mode=0)
   # refold the unfolded tensor
   tl.fold(unfolded, mode=0, shape=tensor.shape)
   
You can as easily manipulate tensors in **decomposed** form:

.. code-block:: python

   tensor = random.random_tucker(shape=(3, 4, 5), rank=(2, 3, 4))
   # We created a tensor of size 3x4x5 in decomposed (Tucker) form with rank (2, 3, 4)

   tl.tucker_tensor.tucker_to_vec(tensor) # Vectorize the tucker tensor

Generally you can manipulate decomposed tensors using the corresponding submodule:
:mod:`tensorly.tucker_tensor`,
:mod:`tensorly.cp_tensor`,
:mod:`tensorly.tt_tensor`,
:mod:`tensorly.tt_matrix`, 
:mod:`tensorly.parafac2_tensor`.


Tensor algebra
--------------

More '*advanced*' tensor algebra functions are located in the aptly named :py:mod:`tensorly.tenalg` module.
This includes for instance, n-mode product, kronecker product, etc.

We now provide a backend system for tensor algebra, which allows to either use our "hand-crafter" implementations
or to dispatch all the operations to einsum. By default, we use the hand-crafted implementations. To switch to einsum, or change the tenalg backend:


.. code-block::

   import tensorly.tenalg as tg

   tg.set_tenalg_backend('core') # This is the default
   tg.kronecker([matrix1, matrix2]) # Hand crafted implementation

   tg.set_tenalg_backend('einsum')
   tg.kronecker([matrix1, matrix2]) # Dispatched to einsum


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

Whether you are training a tensor regression method or combining deep learning and tensor methods, 
you will need metrics to train and assess your method. 
These are implemented in the :py:mod:`tensorly.metrics` module.

Sampling random tensors
-----------------------

To create random tensors, you can use the :py:mod:`tensorly.random` module.
For instance:

.. code-block:: python

   from tensorly import random
   
   # full tensor
   tensor = random.random_tensor((3, 4, 5))
   
   # CP tensor
   tensor = random.random_cp(shape=(3, 4, 5), rank=3)

   # A full tensor with a low-rank CP structure
   tensor = random.random_cp(shape=(3, 4, 5), rank=3, full=True)

Experimental features
---------------------
The module :mod:`tensorly.contrib` contains experimental features.
These are fully tested features, completely integrated in TensorLy but
for which the API or implementation might still be changing.

Currently, this includes tensor-train cross approximation, as well as various 
sparse tensor decompositions (using PyData sparse structures).

Datasets
--------
The :mod:`tensorly.datasets` module contains utility functions for loading and creating data
for testing tensor methods.