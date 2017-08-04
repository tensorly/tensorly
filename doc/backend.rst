TensorLy's backend
==================

TensorLy supports several backends transparently: NumPy and MXNet (the default).

To change the backend, e.g. to NumPy, you can change the value of `default_backend` in tensorly/__init__.
Alternatively during the execution you can call set_backend('numpy').

