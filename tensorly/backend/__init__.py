import warnings
from .core import Backend
import importlib
import os
import sys
import threading
from contextlib import contextmanager
import inspect

_DEFAULT_BACKEND = 'numpy'
_KNOWN_BACKENDS = {'numpy': 'NumpyBackend',
                   'mxnet':'MxnetBackend', 
                   'pytorch':'PyTorchBackend', 
                   'tensorflow':'TensorflowBackend',
                   'cupy':'CupyBackend'}

_LOADED_BACKENDS = {}
_LOCAL_STATE = threading.local()

def initialize_backend():
    """Initialises the backend

    1) retrieve the default backend name from the `TENSORLY_BACKEND` environment variable
        if not found, use _DEFAULT_BACKEND
    2) sets the backend to the retrieved backend name
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
    
    Parameters
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
        global _DEFAULT_BACKEND
        _DEFAULT_BACKEND = backend.backend_name

def get_backend():
    """Returns the name of the current backend
    """
    return _get_backend_method('backend_name')

def _get_backend_method(key):
    try:
        return getattr(_LOCAL_STATE.backend, key)
    except AttributeError:
        return getattr(_LOADED_BACKENDS[_DEFAULT_BACKEND], key)

def _get_backend_dir():
    return [k for k in dir(_LOCAL_STATE.backend) if not k.startswith('_')]

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

def override_module_dispatch(module_name, getter_fun, dir_fun):
    """Override the module's dispatch mechanism

        In Python >= 3.7, we use module's __getattr__ and __dir__
        On older versions, we override the sys.module[__name__].__class__
    """
    if sys.version_info >= (3, 7, 0):
        sys.modules[module_name].__getattr__ = getter_fun
        sys.modules[module_name].__dir__ = dir_fun

    else:
        import types

        class BackendAttributeModuleType(types.ModuleType):
            """A module type to dispatch backend generic attributes."""
            def __getattr__(self, key):
                return getter_fun(key)

            def __dir__(self):
                out = set(super().__dir__())
                out.update({k for k in dir(_LOCAL_STATE.backend) if not k.startswith('_')})
                return list(out)

        sys.modules[module_name].__class__ = BackendAttributeModuleType


def dispatch(method):
    """Create a dispatched function from a generic backend method."""
    name = method.__name__

    def inner(*args, **kwargs):
        return _get_backend_method(name)(*args, **kwargs)

    # We don't use `functools.wraps` here because some of the dispatched
    # methods include the backend (`self`) as a parameter. Instead we manually
    # copy over the needed information, and filter the signature for `self`.
    for attr in ['__module__', '__name__', '__qualname__', '__doc__',
                 '__annotations__']:
        try:
            setattr(inner, attr, getattr(method, attr))
        except AttributeError:
            pass

    sig = inspect.signature(method)
    if 'self' in sig.parameters:
        parameters = [v for k, v in sig.parameters.items() if k != 'self']
        sig = sig.replace(parameters=parameters)
    inner.__signature__ = sig

    return inner

# Generic methods, exposed as part of the public API
context = dispatch(Backend.context)
tensor = dispatch(Backend.tensor)
is_tensor = dispatch(Backend.is_tensor)
shape = dispatch(Backend.shape)
ndim = dispatch(Backend.ndim)
to_numpy = dispatch(Backend.to_numpy)
copy = dispatch(Backend.copy)
concatenate = dispatch(Backend.concatenate)
stack = dispatch(Backend.stack)
reshape = dispatch(Backend.reshape)
transpose = dispatch(Backend.transpose)
moveaxis = dispatch(Backend.moveaxis)
arange = dispatch(Backend.arange)
ones = dispatch(Backend.ones)
zeros = dispatch(Backend.zeros)
zeros_like = dispatch(Backend.zeros_like)
eye = dispatch(Backend.eye)
where = dispatch(Backend.where)
clip = dispatch(Backend.clip)
max = dispatch(Backend.max)
min = dispatch(Backend.min)
argmax = dispatch(Backend.argmax)
argmin = dispatch(Backend.argmin)
all = dispatch(Backend.all)
mean = dispatch(Backend.mean)
sum = dispatch(Backend.sum)
prod = dispatch(Backend.prod)
sign = dispatch(Backend.sign)
abs = dispatch(Backend.abs)
sqrt = dispatch(Backend.sqrt)
norm = dispatch(Backend.norm)
dot = dispatch(Backend.dot)
kron = dispatch(Backend.kron)
solve = dispatch(Backend.solve)
qr = dispatch(Backend.qr)
kr = dispatch(Backend.kr)
partial_svd = dispatch(Backend.partial_svd)


# Initialise the backend to the default one
initialize_backend()
override_module_dispatch(__name__, 
                         _get_backend_method,
                         _get_backend_dir)
