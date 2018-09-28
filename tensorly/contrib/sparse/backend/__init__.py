from importlib import import_module
from contextlib import contextmanager

from ....backend import set_backend, get_backend


_KNOWN_BACKENDS = ('numpy',)
_LOADED_BACKENDS = {}


@contextmanager
def using_sparse_backend():
    backend = get_backend()
    sparse_backend = '%s.sparse' % backend
    if sparse_backend not in _LOADED_BACKENDS:
        # load the backend
        if backend in _KNOWN_BACKENDS:
            import_module('tensorly.contrib.sparse.backend.%s_backend' % backend)
        else:
            raise NotImplementedError("Sparse functionality for backend %r" % backend)
    with set_backend(_LOADED_BACKENDS[sparse_backend]):
        yield


def register_sparse_backend(backend):
    """Register a sparse backend with tensorly

    Parameters
    ----------
    backend : Backend
        The backend to register.
    """
    _LOADED_BACKENDS[backend.backend_name] = backend
