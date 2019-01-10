import warnings
from .core import Backend
import importlib
import os
import sys
import threading
from contextlib import contextmanager

_DEFAULT_BACKEND = 'numpy'
_KNOWN_BACKENDS = {'numpy': 'NumpyBackend', 'mxnet':'MxnetBackend', 
                    'pytorch':'PyTorchBackend', 'tensorflow':'TensorflowBackend',
                    'cupy':'CupyBackend'}
_LOADED_BACKENDS = {}
_LOCAL_STATE = threading.local()

def initialize_backend():
    """Initialises the backend

    1) retrieve the default backend name from the `TENSORLY_BACKEND` environment variable
        if not found, use _DEFAULT_BACKEND
    2) sets the backend to the retrived backend name
    """
    backend_name = os.environ.get('TENSORLY_BACKEND', _DEFAULT_BACKEND)
    if backend_name not in _KNOWN_BACKENDS:
        msg = ("TENSORLY_BACKEND should be one of {}, got {}. Defaulting to {}'").format(
                    ', '.join(map(repr, _KNOWN_BACKENDS)),
                        backend_name, _DEFAULT_BACKEND)
        warnings.warn(msg, UserWarning)
        backend_name = _DEFAULT_BACKEND
    
    set_backend(backend_name, local_threadsafe=False)

def register_backend(backend_name):
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
        module = importlib.import_module('tensorly.backend.{0}_backend'.format(backend_name))
        backend = getattr(module, _KNOWN_BACKENDS[backend_name])()
        _LOADED_BACKENDS[backend_name] = backend
    else:
        msg = "Unknown backend name {0!r}, known backends are [{1}]".format(
                backend_name, ', '.join(map(repr, _KNOWN_BACKENDS)))
        raise ValueError(msg)

def set_backend(backend, local_threadsafe=False):
    """Changes the backend to the specified one
    
    Parameters
    ----------
    backend : tensorly.Backend or str
        name of the backend to load or Backend Class
    local_threadsafe : bool, optional, default is False
        If False, set the backend as default for all threads        
    """
    if not isinstance(backend, Backend):
        # Backend is a string
        if backend not in _LOADED_BACKENDS:
            register_backend(backend)

        backend = _LOADED_BACKENDS[backend]

    # Set the backend
    _LOCAL_STATE.backend = backend
    
    if not local_threadsafe:
        _DEFAULT_BACKEND = backend

def get_backend():
    """Returns the name of the current backend
    """
    return _LOCAL_STATE.backend.backend_name

@contextmanager
def backend_context(backend, local_threadsafe=False):
    """Context manager to set the backend for TensorLy.

    Parameters
    ----------
    backend : {'numpy', 'mxnet', 'pytorch', 'tensorflow', 'cupy'}
        The name of the backend to use. Default is 'numpy'.
    local_threadsafe : bool, optional
        If True, the backend will not become the default backend for all threads.
        Note that this only affects threads where the backend hasn't already
        been explicitly set. If False (default) the backend is set for the
        entire session.

    Examples
    --------
    Set the backend to numpy globally for this thread:

    >>> import tensorly as tl
    >>> tl.set_backend('numpy')
    >>> with tl.backend_context('pytorch'):
    ...     pass
    """
    _old_backend = get_backend()
    set_backend(backend, local_threadsafe=local_threadsafe)
    try:
        yield
    finally:
        set_backend(_old_backend)

# Initialise the backend to the default one
initialize_backend()

# Dynamically dispatch the methods and attributes from the current backend
# Python 3.7 or higher, use module __getattr__ (PEP 562)
if sys.version_info >= (3, 7, 0):
    def __getattr__(item):
        return getattr(_LOCAL_STATE.backend, item)

    def __dir__():
        return [k for k in dir(_LOCAL_STATE.backend) if not k.startswith('_')]

# Python 3.6 or lower: we need to overwrite the class of the module...
else:
    import types

    class BackendAttributeModuleType(types.ModuleType):
        """A module type to dispatch backend generic attributes."""
        def __getattr__(self, key):
            return getattr(_LOCAL_STATE.backend, key)

        def __dir__(self):
            out = set(super(BackendAttributeModuleType, self).__dir__())
            out.update({k for k in dir(_LOCAL_STATE.backend) if not k.startswith('_')})
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
