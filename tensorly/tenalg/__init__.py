""" 
The :mod:`tensorly.tenalg` module contains utilities for Tensor Algebra 
operations such as khatri-rao or kronecker product, n-mode product, etc.
""" 
from contextlib import contextmanager
import os
from functools import wraps
import warnings
from collections import ChainMap

from .core_tenalg import mode_dot, multi_mode_dot
from .core_tenalg import kronecker
from .core_tenalg import khatri_rao
from .core_tenalg import inner
from .core_tenalg import outer, batched_outer
from .core_tenalg import higher_order_moment
from .core_tenalg import _tt_matrix_to_tensor
from .core_tenalg import tensordot

from . import core_tenalg as core
from . import einsum_tenalg

from ..backend import _LOCAL_STATE

_DEFAULT_TENALG_BACKEND = 'core'
_KNOWN_TENALG_BACKENDS = {'core':core,
                          'einsum':einsum_tenalg}
_LOADED_BACKENDS_NAMES = []

# Mapping name: funs are stored here
_LOADED_TENALG_BACKENDS = {}
# Mapping for current backend
_BACKEND_TENALG_MAPPING = dict()
# User specified override
_USER_DEFINED_TENALG_MAPPING = dict()

_LOCAL_STATE._USER_DEFINED_TENALG_MAPPING = dict()
_LOCAL_STATE._BACKEND_TENALG_MAPPING = dict()

dispatched_tenalg_funs = []

def initialize_tenalg_backend():
    """Initialises the backend

    1) retrieve the default backend name from the `TENSORLY_TENALG_BACKEND` environment variable
        if not found, use _DEFAULT_TENALG_BACKEND
    2) sets the backend to the retrieved backend name
    """
    tenalg_backend_name = os.environ.get('TENSORLY_TENALG_BACKEND', _DEFAULT_TENALG_BACKEND)
    if tenalg_backend_name not in _KNOWN_TENALG_BACKENDS:
        msg = ("TENSORLY_BACKEND should be one of {}, got {}. Defaulting to {}'").format(
            ', '.join(map(repr, _KNOWN_TENALG_BACKENDS)),
            tenalg_backend_name, _DEFAULT_TENALG_BACKEND)
        warnings.warn(msg, UserWarning)
        tenalg_backend_name = _DEFAULT_TENALG_BACKEND

    set_tenalg_backend(tenalg_backend_name, local_threadsafe=False)

def get_tenalg_backend():
    """Returns the current backend
    """
    if hasattr(_LOCAL_STATE, 'tenalg_backend'):
        return _LOCAL_STATE.tenalg_backend
    return _DEFAULT_TENALG_BACKEND

def register_tenalg_backend(backend_name):
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
    if backend_name in _KNOWN_TENALG_BACKENDS:
        module = _KNOWN_TENALG_BACKENDS[backend_name]
        # backend = getattr(module, _KNOWN_TENALG_BACKENDS[backend_name])()
        current_mapping = {name:getattr(module, name) for name in dispatched_tenalg_funs if hasattr(module, name)}
        other_backends = [_LOADED_TENALG_BACKENDS[n] for n in _LOADED_TENALG_BACKENDS.keys() if n != backend_name ]
        mapping = ChainMap(_LOCAL_STATE._USER_DEFINED_TENALG_MAPPING, _USER_DEFINED_TENALG_MAPPING, current_mapping, *other_backends)
        _LOADED_TENALG_BACKENDS[backend_name] = {name:mapping[name] for name in dispatched_tenalg_funs if name in mapping}
    else:
        raise ValueError(f'Unknown tenalg backend {backend_name}.')

def set_tenalg_backend(backend_name='core', local_threadsafe=False):
    """Set the current tenalg backend

    Parameters
    ----------
    backend : {'core', 'einsum'}
        * if 'core', our manually optimized implementations are used 
        * if 'einsum', all operations are dispatched to ``einsum``

    If True, the backend will not become the default backend for all threads.
        Note that this only affects threads where the backend hasn't already
        been explicitly set. If False (default) the backend is set for the
        entire session.
    """
    if isinstance(backend_name, dict):
        backend_mapping = backend_name
    else:
        try:
            backend_mapping = _LOADED_TENALG_BACKENDS[backend_name]
        except KeyError:
            register_tenalg_backend(backend_name)
            backend_mapping = _LOADED_TENALG_BACKENDS[backend_name]
    
    _LOCAL_STATE.tenalg_backend = backend_mapping

    if not local_threadsafe:
        global _DEFAULT_TENALG_BACKEND
        global _BACKEND_TENALG_MAPPING
        _DEFAULT_TENALG_BACKEND = backend_name
        _BACKEND_TENALG_MAPPING = backend_mapping

@contextmanager
def tenalg_backend_context(backend, local_threadsafe=False):
    """Context manager to set the backend for TensorLy.

    Parameters
    ----------
    backend : {'core', 'einsum'}
        * if 'core', our manually optimized implementations are used 
        * if 'einsum', all operations are dispatched to ``einsum``
    local_threadsafe : bool, optional
        If True, the backend will not become the default backend for all threads.
        Note that this only affects threads where the backend hasn't already
        been explicitly set. If False (default) the backend is set for the
        entire session.

    Examples
    --------
    Set the backend to numpy globally for this thread:

    >>> import tensorly.tenalg as tlg
    >>> tlg.set_tenalg_backend('core')
    >>> with tlg.set_tenalg_backend('einsum'):
    ...     pass
    """
    _old_backend = get_tenalg_backend()
    set_tenalg_backend(backend, local_threadsafe=local_threadsafe)
    try:
        yield
    finally:
        set_tenalg_backend(_old_backend)

def dynamically_dispatch_tenalg(function):
    name = function.__name__
    global dispatched_tenalg_funs
    dispatched_tenalg_funs.append(name)

    @wraps(function)
    def dynamically_dispatched_fun(*args, **kwargs):
        if hasattr(_LOCAL_STATE, 'tenalg_backend'):
            return _LOCAL_STATE.tenalg_backend[name](*args, **kwargs)
        return _BACKEND_TENALG_MAPPING[name](*args, **kwargs)

    return dynamically_dispatched_fun

mode_dot = dynamically_dispatch_tenalg(mode_dot)
multi_mode_dot = dynamically_dispatch_tenalg(multi_mode_dot)
kronecker = dynamically_dispatch_tenalg(kronecker)
khatri_rao = dynamically_dispatch_tenalg(khatri_rao)
inner = dynamically_dispatch_tenalg(inner)
outer = dynamically_dispatch_tenalg(outer)
batched_outer = dynamically_dispatch_tenalg(batched_outer)
tensordot = dynamically_dispatch_tenalg(tensordot)
higher_order_moment = dynamically_dispatch_tenalg(higher_order_moment)

initialize_tenalg_backend()
