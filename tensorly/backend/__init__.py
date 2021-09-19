import warnings
from numpy.lib.function_base import disp

from scipy.sparse.linalg import dsolve
from tensorly.backend.core import Backend
import importlib
import os
import sys
import threading
from contextlib import contextmanager
import inspect
import copy
from collections import ChainMap
import tensorly as tl

import tensorly.backend.pytorch_backend as pt

# These store the global variables shared accross threads
_DEFAULT_BACKEND = 'numpy'
_KNOWN_BACKENDS = {'numpy': 'NumpyBackend',
                   'mxnet': 'MxnetBackend',
                   'pytorch': 'PyTorchBackend',
                   'tensorflow': 'TensorflowBackend',
                   'cupy': 'CupyBackend',
                   'jax': 'JaxBackend'}
# Mapping name: funs are stored here
_LOADED_BACKENDS = {}
# Mapping for current backend
_BACKEND_FUNS = dict()
# User specified override
_USER_DEFINED_FUNS = dict()

# Thread-safe variables: stores local backend mappings
_LOCAL_STATE = threading.local()
_LOCAL_STATE._USER_DEFINED_FUNS = dict()
_LOCAL_STATE._BACKEND_FUNS = dict()


dispatched_functions = ['reshape', 'moveaxis', 'any', 'trace', 'shape', 'ndim',
                        'where', 'copy', 'transpose', 'arange', 'ones', 'zeros',
                        'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min', 'matmul',
                        'all', 'mean', 'sum', 'cumsum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
                        'argmax', 'stack', 'conj', 'diag', 'einsum', 'log2', 'dot', 'tensordot', 
                        'sin', 'cos',
                        'solve', 'qr', 'svd', 'eigh', 'randn', 'check_random_state',
                        'index_update', 'context', 'tensor', 'norm'
                       ]
dispatched_attributes = ['int64', 'int32', 'float64', 'float32', 
                         'complex128', 'complex64', 'SVD_FUNS', 'index']


def initialize_backend():
    """Initializes the backend by creating the function mapping

    1) retrieve the default backend name from the `TENSORLY_BACKEND` environment variable
        if not found, use _DEFAULT_BACKEND
    2) sets the backend to the retrieved backend name
    """
    backend_name = os.environ.get('TENSORLY_BACKEND', _DEFAULT_BACKEND)
    if backend_name not in _KNOWN_BACKENDS:
        msg = (f"TENSORLY_BACKEND should be one of {_KNOWN_BACKENDS.keys()}, but got {backend_name}."
               f"Defaulting to the default: '{_DEFAULT_BACKEND}'")
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
        mapping = {name:getattr(backend, name) for name in dispatched_functions+dispatched_attributes}
        _LOADED_BACKENDS[backend_name] = mapping
    else:
        msg = "Unknown backend name {0!r}, known backends are [{1}]".format(
            backend_name, ', '.join(map(repr, _KNOWN_BACKENDS)))
        raise ValueError(msg)

def set_backend(backend_name, local_threadsafe=False):
    """Changes the backend to the specified one
    
    Parameters
    ----------
    backend : tensorly.Backend or str
        name of the backend to load or Backend Class
    local_threadsafe : bool, optional, default is False
        If False, set the backend as default for all threads        
    """
    try:
        backend_funs = _LOADED_BACKENDS[backend_name]
    except KeyError:
        register_backend(backend_name)
        backend_funs = _LOADED_BACKENDS[backend_name]
    
    lookup = ChainMap(_LOCAL_STATE._USER_DEFINED_FUNS, _USER_DEFINED_FUNS, backend_funs)
    for name in dispatched_functions:
        _LOCAL_STATE._BACKEND_FUNS[name] = lookup[name]
    for name in dispatched_attributes:
        _LOCAL_STATE._BACKEND_FUNS[name] = lookup[name]

    if not local_threadsafe:
        global _DEFAULT_BACKEND
        global _BACKEND_FUNS
        _DEFAULT_BACKEND = backend_name
        _BACKEND_FUNS = _LOCAL_STATE._BACKEND_FUNS

def get_backend():
    """Returns the name of the current backend
    """
    return _get_backend_method('backend_name')

def _get_backend_method(key):
    return _LOCAL_STATE._BACKEND_FUNS.get(key, _BACKEND_FUNS.get(key))

def _get_backend_dir():
    return dispatched_functions + dispatched_attributes
    #[k for k in dir(_LOCAL_STATE.backend) if not k.startswith('_')]

def dispatch(method):
    """Create a dispatched function from a generic backend method."""
    name = method.__name__

    def dynamically_dispatched_method(*args, **kwargs):
        return _BACKEND_FUNS.get(name, _LOCAL_STATE._BACKEND_FUNS.get(name))(*args, **kwargs)
#         return _get_backend_method(name)(*args, **kwargs)

    # We don't use `functools.wraps` here because some of the dispatched
    # methods include the backend (`self`) as a parameter. Instead we manually
    # copy over the needed information, and filter the signature for `self`.
    for attr in ['__module__', '__name__', '__qualname__', '__doc__',
                 '__annotations__']:
        try:
            setattr(dynamically_dispatched_method, attr, getattr(method, attr))
        except AttributeError:
            pass

    sig = inspect.signature(method)
    if 'self' in sig.parameters:
        parameters = [v for k, v in sig.parameters.items() if k != 'self']
        sig = sig.replace(parameters=parameters)
    dynamically_dispatched_method.__signature__ = sig

    return dynamically_dispatched_method

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

for name in dispatched_functions:
    exec(f'{name} = dispatch(Backend.{name})')

# Initialise the backend to the default one
initialize_backend()

# dispatch non-callables (e.g. dtypes, index)
override_module_dispatch(__name__, 
                         _get_backend_method,
                         _get_backend_dir)
