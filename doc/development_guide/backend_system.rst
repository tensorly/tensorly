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

How the Backend System Works
----------------------------
When a backend is set, the backend specific module (called
``tensorly.backend.{name}_backend``) is dynamically imported (via ``importlib``)
if the backend has not already been loaded. Once the backend is loaded, it is
grabbed from an internal cache (called ``tensorly.backend.core._LOADED_BACKENDS``)
and set as the current backend (called ``tensorly.backend.core._STATE.backend``).

The backends themselves are instances of the ``tensorly.backend.core.Backend`` class
or subclasses, which implement ``staticmethods`` of the common TensorLy API
(e.g. ``tensor``, ``fold``, ``norm``, etc.). These backend classes are added to the
``_LOADED_BACKENDS`` cache on import of the backend module
(``tensorly.backend.{name}_backend``) via the ``register_backend`` functions.

The TensorLy API functions are then dynamically farmed out to the backend
staticmethods via the dispatching mechanism provided by
``tensorly.backend.core.dispatch``. This ensure that the API function is
wrapped such that it has the correct docstring, name, function signature, and
other required minutia.

Additionally, the modules themselves are also wrapped such that certain
global (module-level) variables ar dynamically dispatched as well, like
``property`` or other class descriptors. This is to ensure that variables
such as ``int32`` or ``float64`` point to the correct, backend-specific
object.  For example, numpy's ``np.int32`` and tensorflow's ``tf.int32``
are not compatible. The dynamic dispatch of these module-level varaibles
is implemented in the ``tensorly.backend.core.BackendAttributeModuleType``
class.