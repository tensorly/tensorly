TensorLy's backend
==================

Backend?
--------
To represent tensors and for numerical computation, TensorLy supports several backends transparently: NumPy (the default), MXNet, and PyTorch.
For the end user, the interface is exactly the same, but under the hood, a different library is used to represent multi-dimensional arrays and perform computations on these.

Why backends
------------
The goal of TensorLy is to make tensor methods accessible. While NumPy needs no introduction, the MXNet and PyTorch backends are especially useful as they allows to perform transparently computation on CPU or GPU. Last but not least, using MXNet or PyTorch as a backend, we are able to combine tensor methods and deep learning easily!

How do I change the backend?
----------------------------
To change the backend, e.g. to NumPy, you can change the value of ``default_backend`` in tensorly/__init__.
Alternatively during the execution, assuming you have imported TensorLy as ``import tensorly as tl``, you can change the backend in your code by calling ``tl.set_backend('numpy')``.

