import warnings
from .core import Backend
import importlib
import inspect
import os
import threading
from contextlib import contextmanager

class BackendManager():
    _DEFAULT_BACKEND = 'numpy'
    _KNOWN_BACKENDS = {'numpy': 'NumpyBackend', 'mxnet':'MxnetBackend', 
                       'pytorch':'PyTorchBackend', 'tensorflow':'TensorflowBackend',
                       'cupy':'CupyBackend'}
    _LOADED_BACKENDS = {}
    _BACKEND = threading.local()

    @staticmethod
    def init():
        """Initialises the backend

        1) retrieve the default backend name from the `TENSORLY_BACKEND` environment variable
            if not found, use BackendManager._DEFAULT_BACKEND
        2) sets the backend to the retrived backend name
        """
        backend_name = os.environ.get('TENSORLY_BACKEND', BackendManager._DEFAULT_BACKEND)
        if backend_name not in BackendManager._KNOWN_BACKENDS:
            msg = ("TENSORLY_BACKEND should be one of {}, got {}. Defaulting to {}'").format(
                        ', '.join(map(repr, BackendManager._KNOWN_BACKENDS)),
                         backend_name, BackendManager._DEFAULT_BACKEND)
            warnings.warn(msg, UserWarning)
            backend_name = BackendManager._DEFAULT_BACKEND
        
        BackendManager.set_backend(backend_name, local_threadsafe=False)

    @staticmethod
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
             in `BackendManager._KNOWN_BACKEND`
        """

        if backend_name in BackendManager._KNOWN_BACKENDS:
            module = importlib.import_module('tensorly.backend.{0}_backend'.format(backend_name))
            backend = getattr(module, BackendManager._KNOWN_BACKENDS[backend_name])()
            BackendManager._LOADED_BACKENDS[backend_name] = backend
        else:
            msg = "Unknown backend name {0!r}, known backends are [{1}]".format(
                    backend_name, ', '.join(map(repr, BackendManager._KNOWN_BACKENDS)))
            raise ValueError(msg)
    
    @staticmethod
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
            if backend not in BackendManager._LOADED_BACKENDS:
                BackendManager.register_backend(backend)

            backend = BackendManager._LOADED_BACKENDS[backend]

        # Set the backend
        BackendManager._BACKEND.backend = backend
        
        if not local_threadsafe:
            BackendManager._DEFAULT_BACKEND = backend
    
    @staticmethod
    def get_backend():
        """Returns the name of the current backend
        """
        return BackendManager._BACKEND.backend.backend_name

    @staticmethod
    def __repr__():
        return 'tensorly.BackendManager (backend={0!r})'.format(
            BackendManager._BACKEND.backend.backend_name)

get_backend = BackendManager.get_backend
set_backend = BackendManager.set_backend

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
    _old_backend = BackendManager.get_backend()
    BackendManager.set_backend(backend, local_threadsafe=local_threadsafe)
    try:
        yield
    finally:
        BackendManager.set_backend(_old_backend)

# Initialise the backend to the default one
BackendManager.init()

# Dynamically dispatch the methods and attributes from the current backend
def __getattr__(item):
    return getattr(BackendManager._BACKEND.backend, item)

def __dir__():
    return [k for k in dir(BackendManager._BACKEND.backend) if not k.startswith('_')]
