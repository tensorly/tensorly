TensorLy's backend
==================

Backend?
--------
To represent tensors and for numerical computation, TensorLy supports several backends transparently: NumPy and MXNet (the default).
For the end user, the interface is exactly the same, but under the hood, a different library is used to represent multi-dimensional arrays and perform computations on these.

Why backends
------------
The goal of TensorLy is to make tensor methods accessible. While NumPy needs no introduction, the MXNet backend is especially useful as it allows to perform transparently computation on CPU or GPU.
Last but not least, using the MXNet backend, we are able to combine tensor methods and deep learning easily!

How do I change the backend?
----------------------------
To change the backend, e.g. to NumPy, you can change the value of `default_backend` in tensorly/__init__.
Alternatively during the execution you can call set_backend('numpy').

