.. _user_guide-backend:

TensorLy's backend system
=========================

.. note::

   In short, you can write your code using TensorLy and you can transparently combine it and execute with any of the backends. 
   Currently we support NumPy PyTorch, MXNet, JAX, TensorFlow and CuPy as backends.


Backend?
--------
To represent tensors and for numerical computation, TensorLy supports several backends transparently: the ubiquitous NumPy (the default), MXNet, and PyTorch.
For the end user, the interface is exactly the same, but under the hood, a different library is used to represent multi-dimensional arrays and perform computations on these.

In other words, you write your code using TensorLy and can then decide whether the computation is done using NumPy, PyTorch or MXNet.

Why backends?
-------------
The goal of TensorLy is to make tensor methods accessible.
While NumPy needs no introduction, other backends such as MXNet and PyTorch backends are especially useful as they allows to perform transparently computation on CPU or GPU. 
Last but not least, using MXNet or PyTorch as a backend, we are able to combine tensor methods and deep learning easily!



How do I change the backend?
----------------------------
To change the backend, e.g. to NumPy, you can change the value of ``default_backend`` in tensorly/__init__.
Alternatively during the execution, assuming you have imported TensorLy as ``import tensorly as tl``, you can change the backend in your code by calling ``tl.set_backend('numpy')``.

.. important::
   
   NumPy is installed by default with TensorLy if you haven't already installed it. 
   However, to keep dependencies as minimal as possible, and to not complexify installation, neither MXNet nor PyTorch are installed.  If you want to use them as backend, you will have to install them first. 
   It is easy however, simply refer to their respective installation instructions:

   * `PyTorch <http://pytorch.org>`_
   * `MXNet <https://mxnet.apache.org/install/index.html>`_
   * `JAX <https://jax.readthedocs.io/en/latest/developer.html#building-or-installing-jaxlib>`_ 
   * `CuPy <https://docs.cupy.dev/en/stable/install.html>`_
   * `TensorFlow <https://www.tensorflow.org/install>`_ 


Once you change the backend, all the computation is done using that backend.

Context of a tensor
-------------------

Different backends have different parameters associated with the tensors. For instance, in NumPy we traditionally set the dtype when creating an ndarray, while in mxnet we also have to change the *context* (GPU or CPU), with the `ctx` argument. Similarly, in PyTorch, we might want to create a FloatTensor for CPU and a cuda.FloatTensor for GPU. 

To handle this difference, we implemented a `context` function, that, given a tensor, returns a dictionary of values characterising that tensor. A function getting a tensor as input and creating a new tensor should use that context to create the new tensor.

For instance:

.. code-block:: python
  
   import tensorly as tl

   def trivial_fun(tensor):
      """ Trivial function that takes a tensor and create a new one
            with value tensor + 2...
      """
      # context is a dict of values
      context = tl.context(tensor)
      # when creating a new tensor we use these as parameters
      new_tensor = tl.tensor(tensor + 2, **context)
      return new_tensor

Basic functions
---------------
We have isolated the basic functions required for tensor methods in the backend, and provide a uniform API using wrappers when necessary.
In practice, this means that function like `min`, `max`, `reshape`, etc, are accessible from the backend:

.. code-block:: python

   import tensorly as tl
   import numpy as np

   tl.set_backend('pytorch') # or any other backend

   tensor = tl.tensor(np.random.random((10, 10, 10)))

   # This will call the correct function depending on the backend
   min_value = tl.min(tensor)
   
   unfolding = tl.unfold(tensor, mode=0)
   U, S, V = tl.truncated_svd(unfolding, n_eigenvecs=5)

This will allow your code to work transparently with any of the backend.


Case study: TensorLy and PyTorch
--------------------------------

Let's go through the creation and decomposition of a tensor, using PyTorch.

On CPU
++++++

First, we import tensorly and set the backend:

.. code:: python

   import tensorly as tl
   tl.set_backend('pytorch')

Now, let's create a random tensor using the :mod:`tensorly.random` module:

.. code:: python

   from tensorly import random

   tensor = random.random_tensor((10, 10, 10))
   # tensor is a PyTorch Tensor!

We can decompose it easily, here using a Tucker decomposition: 
First, we create a decomposition instance, which keeps the number of parameters the same
and with a random initialization. We then fit it to our tensor.

.. code:: python

   from tensorly.decomposition import Tucker

   decomp = Tucker(rank='same', init='random')
   cp_tensor = decomp.fit_transform(tensor)

You can reconstruct the full tensor and measure the reconstruction error:

.. code:: python

   rec = cp_tensor.to_tensor()
   error = tl.norm(tensor - rec)/tl.norm(tensor)

On GPU
++++++
Now, imaging you want everything to run on GPU: this is very easy using TensorLy and the PyTorch backend: 
you simply send the tensor to the GPU!

There are to main ways to do this: either you specify the context during the creation of the tensor
or you use pytorch tensors' properties to send them to the desired device post-creation.

.. code:: python

   # Specify context during creation
   tensor = random.random_tensor(shape=(10, 10, 10), device='cuda', dtype=tl.float32)

   # Posthoc 
   tensor = random.random_tensor(shape=(10, 10, 10))
   tensor = tensor.to('cuda')

The rest is exactly the same, nothing more to do!

.. code:: python

   decomp = Tucker(rank='same', init='random')
   cp_tensor = decomp.fit_transform(tensor) # Runs on GPU!


Using static dispatching
------------------------

We optimized the dynamical dispatch so the overhead is negligeable. 
However, if you only want to use one backend, you can first set it and then switch to static dispatching:

>>> tl.use_static_dispatch()

And you can switch back to dynamical dispatching just as easily:

>>> tl.use_dynamic_dispatch()

