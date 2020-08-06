""" 
The :mod:`tensorly.tenalg` module contains utilities for Tensor Algebra 
operations such as khatri-rao or kronecker product, n-mode product, etc.
""" 
from contextlib import contextmanager

from functools import wraps
import warnings

from .core_tenalg import mode_dot, multi_mode_dot
from .core_tenalg import kronecker
from .core_tenalg import khatri_rao
from .core_tenalg import inner, outer
from .core_tenalg import contract
from .core_tenalg import tensor_dot, batched_tensor_dot
from .core_tenalg import higher_order_moment

from . import core_tenalg as core
from . import einsum_tenalg

from ..backend import _LOCAL_STATE

_DEFAULT_TENALG_BACKEND = 'core'
_LOCAL_STATE.tenalg_backend = _DEFAULT_TENALG_BACKEND

_BACKENDS = {'core':core,
             'einsum':einsum_tenalg}


def get_tenalg_backend():
    return _LOCAL_STATE.tenalg_backend

def set_tenalg_backend(backend='core', local_threadsafe=False):
    if backend in _BACKENDS:
        _LOCAL_STATE.tenalg_backend = backend
        if local_threadsafe == False:
            global _DEFAULT_TENALG_BACKEND
            _DEFAULT_TENALG_BACKEND = backend
    else:
        raise ValueError(f'Unknown tenalg backend {backend}')

@contextmanager
def tenalg_backend_context(backend, local_threadsafe=False):
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
    _old_backend = get_tenalg_backend()
    set_tenalg_backend(backend, local_threadsafe=local_threadsafe)
    try:
        yield
    finally:
        set_tenalg_backend(_old_backend)

def dynamically_dispatch_tenalg(function):
    name = function.__name__

    @wraps(function)
    def dynamically_dispatched_fun(*args, **kwargs):
        #print('hello')
        current_backend = _BACKENDS[_LOCAL_STATE.tenalg_backend]
        if hasattr(current_backend, name):
            fun = getattr(current_backend, name)(*args, **kwargs)
        else:
            warnings.warn(f'tenalg: defaulting to core tenalg backend, {name}'
                          f'not yet implemented in {_LOCAL_STATE.tenalg_backend} backend.')
            fun = getattr(core, name)(*args, **kwargs)
        return fun
    return dynamically_dispatched_fun


mode_dot = dynamically_dispatch_tenalg(mode_dot)
multi_mode_dot = dynamically_dispatch_tenalg(multi_mode_dot)
kronecker = dynamically_dispatch_tenalg(kronecker)
khatri_rao = dynamically_dispatch_tenalg(khatri_rao)
inner = dynamically_dispatch_tenalg(inner)
contract = dynamically_dispatch_tenalg(contract)
outer = dynamically_dispatch_tenalg(outer)
tensor_dot = dynamically_dispatch_tenalg(tensor_dot)
batched_tensor_dot = dynamically_dispatch_tenalg(batched_tensor_dot)
higher_order_moment = dynamically_dispatch_tenalg(higher_order_moment)

set_tenalg_backend(_DEFAULT_TENALG_BACKEND, local_threadsafe=False)
