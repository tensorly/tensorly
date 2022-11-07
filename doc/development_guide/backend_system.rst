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

Additionally,  we provide a context manager ``backend_context``  
for convenience, whcih may be used to
safely use a backend only for limited context::

    >>> with tl.backend_context('pytorch'):
    ...     pass

This is also thread-safe. The context manager approach is useful in
third-party libraries that wish to ensure a particular backend is used,
no matter what the global backend is set to at the time the library code
is executed.

How the Backend System Works
----------------------------

The Backend class
~~~~~~~~~~~~~~~~~
A backend is represented as a class, which implements the various functions needed
(e.g. `transpose`, `clip`, etc).

A base class ``tensorly.backend.core.Backend`` is given in `tensorly/backend/core.py`,
which implement ``staticmethods`` of the common TensorLy API
(e.g. ``tensor``, ``fold``, ``norm``, etc.).
It will also provide some useful functions by default (e.g. `kron`, `kr`).

This base class should be subclassed when defining a new backend.

Loading a backend 
~~~~~~~~~~~~~~~~~
The logic for loading is in `tensorly/backend/__init__.py`.
A cache of already loaded backend is maintained 
as in a dictionary ``_LOADED_BACKENDS``
(the keys of which are the backends' names and the values the actual backend classes).

When a backend is set (by calling ``tensorly.backend.set_backend``), 
the backend specific module (called``tensorly.backend.{name}_backend``) 
is loaded if the backend has not already been loaded and set as the current backend.

If the backend name is not in ``_LOADED_BACKENDS``,
the corresponding backend module 
(``tensorly.backend.{name}_backend``) 
is dynamically imported 
(via ``importlib``) by the the ``tensorly.backend.register_backend`` function.

Once the backend is loaded, it is
grabbed from the internal cache (``tensorly.backend.core._LOADED_BACKENDS``)
and set as the current backend (``tensorly.backend.core._STATE.backend``).
Note that ``tensorly.backend.core._STATE.backend`` is a thread-local storage.


Backend function's dispatching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
are not compatible. 


The dynamic dispatch of these module-level varaibles
is implemented by the ``tensorly.backend.override_module_dispatch``
function.
This is done in two ways:

* **For Python >= 3.7.0**: using module's _getattr__ and __dir__ as defined by 
   `PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_ 
* **For Python < 3.7.0**: by overwriting 
    sys.modules[module_name].__class__ with a custom class, 
    subclassing ``types.ModuleType`` for which we have overrridden the 
    ``__getitem__`` and ``__dir__`` methods.