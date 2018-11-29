.. _backend_system:

Backend System
==============
The TensorLy backend system allows for switching between multiple backends in
a thread-local way.  You can obtain the back that is currently being used with the
``get_backend()`` function::

    >>> import tensorly as tl
    >>> tl.get_backend()
    'numpy'

You may set the backend globally with the ``set_backend()`` function::

    >>> tl.set_backend('tensorflow')
    >>> tl.get_backend()
    'tensorflow'

Setting the backend is local to the thread that ``set_backend()`` is
executed on. This enables third-party packages (e.g. Dask) to parallelize
operations over multiple backends, as needed. Threads inherit the backend
from the thread that spawned them (which is typically the main thread).
Globally setting the backend supports interactive usage.

Additionally, ``set_backend()`` may be used as a context manager to
safely use a backend only for limited context::

    >>> with tl.set_backend('pytorch'):
    ...     pass

This is also thread-safe. The context manager approach is useful in
third-party libraries that wish to ensure a particular backend is used,
no matter what the global backend is set to at the time the library code
is executed.

