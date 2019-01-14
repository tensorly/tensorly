import importlib
from contextlib import contextmanager

from ....backend import backend_context, get_backend, override_module_dispatch


_KNOWN_BACKENDS = {'numpy': 'NumpySparseBackend'}
_LOADED_BACKENDS = {}

@contextmanager
def sparse_context():
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


override_module_dispatch(__name__, get_backend_method, get_backend_dir)