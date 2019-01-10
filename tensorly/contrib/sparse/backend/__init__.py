import importlib
from contextlib import contextmanager

from ....backend import backend_context, get_backend


_KNOWN_BACKENDS = {'numpy': 'NumpySparseBackend'}
_LOADED_BACKENDS = {}

@contextmanager
def using_sparse_backend():
    backend_name = get_backend()
    if backend_name not in _LOADED_BACKENDS:
        register_sparse_backend(backend_name)

    with backend_context(_LOADED_BACKENDS[backend_name]):
        yield


def register_sparse_backend(backend_name):
    """Registers a new backend by importing the corresponding module 
        and adding the correspond `Backend` class in Backend._LOADED_BACKEND
        under the key `backend_name`
    
    Parameterss
    ----------
    backend_name : str, name of the backend to load
    
    Raises
    ------
    ValueError
        If `backend_name` does not correspond to one listed
            in `_KNOWN_BACKEND`
    """

    if backend_name in _KNOWN_BACKENDS:
        module = importlib.import_module('tensorly.contrib.sparse.backend.{0}_backend'.format(backend_name))
        backend = getattr(module, _KNOWN_BACKENDS[backend_name])()
        _LOADED_BACKENDS[backend_name] = backend
    else:
        msg = "Unknown backend name {0!r}, known backends are [{1}]".format(
                backend_name, ', '.join(map(repr, _KNOWN_BACKENDS)))
        raise ValueError(msg)

def get_backend_method(method_name):
    backend_name = get_backend()
    if backend_name not in _LOADED_BACKENDS:
        register_sparse_backend(backend_name)
    return getattr(_LOADED_BACKENDS[backend_name], method_name)

def get_backend_dir():
    backend_name = get_backend()
    if backend_name not in _LOADED_BACKENDS:
        register_sparse_backend(backend_name)

    return [k for k in dir(_LOADED_BACKENDS[backend_name]) if not k.startswith('_')]


import sys 

# Dynamically dispatch the methods and attributes from the current backend
# Python 3.7 or higher, use module __getattr__ (PEP 562)
if sys.version_info >= (3, 7, 0):
    def __getattr__(item):
        return get_backend_method(item)

    def __dir__():
        return get_backend_dir()

# Python 3.6 or lower: we need to overwrite the class of the module...
else:
    import types

    class BackendAttributeModuleType(types.ModuleType):
        """A module type to dispatch backend generic attributes."""
        def __getattr__(self, key):
            return get_backend_method(item)

        def __dir__(self):
            out = set(super(BackendAttributeModuleType, self).__dir__())
            out.update(get_backend_dir())
            return list(out)

    def _wrap_module(module_name):
        """Wrap a module to dynamically dispatch attributes to the backend.
        Intended use is
        >>> tl.wrap_module(__name__)

        This will effectively overwrite the __getattr__ and __dir__ methods
        to dynamically fetch from the current backend rather than the one set at import time
        """
        sys.modules[module_name].__class__ = BackendAttributeModuleType

    _wrap_module(__name__)
