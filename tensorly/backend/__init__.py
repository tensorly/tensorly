import warnings
from .core import Backend
import importlib
import inspect
import os

class BackendManager():
    _DEFAULT_BACKEND = 'numpy'
    _KNOWN_BACKENDS = {'numpy': 'NumpyBackend', 'mxnet':'MxnetBackend', 
                       'pytorch':'PyTorchBackend', 'tensorflow':'TensorflowBackend',
                       'cupy':'CupyBackend'}
    _LOADED_BACKENDS = {}

    @staticmethod
    def initialise_backend():
        backend = os.environ.get('TENSORLY_BACKEND', BackendManager._DEFAULT_BACKEND)
        if backend not in BackendManager._KNOWN_BACKENDS:
            msg = ("TENSORLY_BACKEND should be one of {%s}, got %r. Defaulting to "
                   "'%r'") % (', '.join(map(repr, BackendManager._KNOWN_BACKENDS)), backend, BackendManager._DEFAULT_BACKEND)
            warnings.warn(msg, UserWarning)
            backend = BackendManager._DEFAULT_BACKEND
        
        BackendManager.set_backend(backend, local_threadsafe=False)
    
    @staticmethod
    def set_backend(backend, local_threadsafe=False):
        if not isinstance(backend, Backend):
            if backend not in BackendManager._LOADED_BACKENDS:
                # load the backend
                if backend in BackendManager._KNOWN_BACKENDS:
                    print(f'Importing backend {backend}')
                    importlib.import_module('tensorly.backend.{0}_backend'.format(backend))
                    #backend = getattr(module, BackendManager._KNOWN_BACKENDS[backend])()
                    #BackendManager._LOADED_BACKENDS[backend.backend_name] = backend
                else:
                    msg = "Unknown backend {0!r}, known backends are {{{1}}}".format(
                            backend, ', '.join(map(repr, BackendManager._KNOWN_BACKENDS)))
                    raise ValueError(msg)

        backend = BackendManager._LOADED_BACKENDS[backend]

        # Set the backend
        BackendManager.backend = backend
        
        if not local_threadsafe:
            BackendManager._DEFAULT_BACKEND = backend
    
    @staticmethod
    def __repr__():
        return 'tensorly.BackendManager (backend={0!r})'.format(BackendManager.backend.backend_name)

    @staticmethod  
    def register_backend(backend):
        """Register a backend with tensorly

        Parameters
        ----------
        backend : Backend
            The backend to register.
        """
        BackendManager._LOADED_BACKENDS[backend.backend_name] = backend

    @staticmethod
    def get_backend():
        return BackendManager.backend.backend_name


BackendManager.initialise_backend()

def __getattr__(item):
    return getattr(BackendManager.backend, item)

def __dir__():
    return [k for k in dir(BackendManager.backend) if not k.startswith('_')]
