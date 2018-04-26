.. _user_guide-backend:

TensorLy's backend system
=========================

Backend?
--------
To represent tensors and for numerical computation, TensorLy supports several backends transparently: the ubiquitous NumPy (the default), MXNet, and PyTorch.
For the end user, the interface is exactly the same, but under the hood, a different library is used to represent multi-dimensional arrays and perform computations on these. In other words, you write your code using TensorLy and can then decide whether the computation is done using NumPy, PyTorch or MXNet.

Why backends?
-------------
The goal of TensorLy is to make tensor methods accessible. While NumPy needs no introduction, the MXNet and PyTorch backends are especially useful as they allows to perform transparently computation on CPU or GPU. Last but not least, using MXNet or PyTorch as a backend, we are able to combine tensor methods and deep learning easily!

How do I change the backend?
----------------------------
To change the backend, e.g. to NumPy, you can change the value of ``default_backend`` in tensorly/__init__.
Alternatively during the execution, assuming you have imported TensorLy as ``import tensorly as tl``, you can change the backend in your code by calling ``tl.set_backend('numpy')``.

.. important::
   
   NumPy is installed by default with TensorLy if you haven't already installed it. 
   However, to keep dependencies as minimal as possible, and to not complexify installation, neither MXNet nor PyTorch are installed.  If you want to use them as backend, you will have to install them first. It is easy however, simply refer to their respective installation instructions:

   * For PyTorch: http://pytorch.org
   * For MXNet https://mxnet.apache.org/install/index.html

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
   U, S, V = tl.partial_svd(unfolding, n_eigenvecs=5)

This will allow your code to work transparently with any of the backend.
