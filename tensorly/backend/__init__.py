import warnings

from .core import Backend
import importlib
import os
import threading
from contextlib import contextmanager
import inspect
import types
import sys


class dynamically_dispatched_class_attribute(object):
    __slots__ = ['name']

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, cls=None):
        if isinstance is None:
            return getattr(cls.current_backend(), self.name)
        else:
            return getattr(instance.current_backend(), self.name)

class BackendManager(types.ModuleType):
    _functions = ['reshape', 'moveaxis', 'any', 'trace', 'shape', 'ndim',
                  'where', 'copy', 'transpose', 'arange', 'ones', 'zeros',
                  'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min', 'matmul',
                  'all', 'mean', 'sum', 'cumsum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
                  'argmax', 'stack', 'conj', 'diag', 'einsum', 'log2', 'dot', 'tensordot', 
                  'sin', 'cos', 'clip', 'kr', 'kron', 'partial_svd', 'lstsq', 'eps', 'finfo',
                  'solve', 'qr', 'randn', 'check_random_state', 'sort', 'eigh',
                  'index_update', 'context', 'tensor', 'norm', 'to_numpy', 'is_tensor',
                  'randomized_range_finder', 'randomized_svd', 'argsort', 'flip', 'count_nonzero'
                 ]
    _attributes = ['int64', 'int32', 'float64', 'float32', 
                   'complex128', 'complex64', 'SVD_FUNS', 'index', 'backend_name']
    available_backend_names = ['numpy', 'mxnet', 'pytorch', 'tensorflow', 'cupy', 'jax']
    _default_backend = 'numpy'
    _loaded_backends = dict()
    _backend = None
    _THREAD_LOCAL_DATA = threading.local()
    _ENV_DEFAULT_VAR = 'TENSORLY_BACKEND'

    @classmethod
    def use_dynamic_dispatch(cls):
        # Define class methods and attributes that dynamically dispatch to the backend
        for name in cls._functions:
            if hasattr(cls, name):
                delattr(cls, name)
            setattr(cls, name, staticmethod(cls.dispatch_backend_method(name, getattr(cls.current_backend(), name))))
        for name in cls._attributes:
            if hasattr(cls, name):
                delattr(cls, name)
            setattr(cls, name, dynamically_dispatched_class_attribute(name))

    @classmethod
    def use_static_dispatch(cls):
        # Define class methods and attributes that dynamically dispatch to the backend
        for name in cls._functions:
            setattr(cls, name, staticmethod(getattr(cls.current_backend(), name)))
        for name in cls._attributes:
            setattr(cls, name, getattr(cls.current_backend(), name))

    @classmethod
    def current_backend(cls):
        """Returns the currently used backend instance

        Returns
        -------
        backend : tensorly.backend.Backend
            Backend instance currently in use
        """
        return cls._THREAD_LOCAL_DATA.__dict__.get('backend', cls._backend)

    @classmethod
    def get_backend(cls):
        """Returns the *name* (str) of the currently used backend
        
        Returns
        -------
        name : str
        """
        return cls._THREAD_LOCAL_DATA.__dict__.get('backend', cls._backend).backend_name

    @classmethod
    def get_backend_dir(cls):
        return cls._attributes + cls._functions

    @classmethod
    def dispatch_backend_method(cls, name, method):
        """Create a dispatched function from a generic backend method."""
        
        def wrapped_backend_method(*args, **kwargs):
            return getattr(cls._THREAD_LOCAL_DATA.__dict__.get('backend', cls._backend), name)(*args, **kwargs)

        # We don't use `functools.wraps` here because some of the dispatched
        # methods include the backend (`cls`) as a parameter. Instead we manually
        # copy over the needed information, and filter the signature for `cls`.
        for attr in ['__module__', '__name__', '__qualname__', '__doc__',
                     '__annotations__']:
            try:
                setattr(wrapped_backend_method, attr, getattr(method, attr))
            except AttributeError:
                pass
    
        getattr(wrapped_backend_method, '__dict__').update(getattr(method,  '__dict__', {}))
        wrapped_backend_method.__wrapped__ = method
        try:
            sig = inspect.signature(method)
            if 'self' in sig.parameters:
                parameters = [v for k, v in sig.parameters.items() if k != 'self']
                sig = sig.replace(parameters=parameters)
            wrapped_backend_method.__signature__ = sig
        except ValueError:
            # If it doesn't have a signature we don't need to remove self
            # This happens for NumPy (e.g. np.where) where inspect.signature(np.where) errors:
            # ValueError: no signature found for builtin <built-in function where>
            pass

        return wrapped_backend_method

    @classmethod
    @contextmanager
    def backend_context(cls, backend, local_threadsafe=False):
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
        _old_backend = cls.current_backend()
        cls.set_backend(backend, local_threadsafe=local_threadsafe)
        try:
            yield
        finally:
            cls.set_backend(_old_backend)

    @classmethod
    def initialize_backend(cls):
        """Initialises the backend

        1) retrieve the default backend name from the `TENSORLY_BACKEND` environment variable
            if not found, use _DEFAULT_BACKEND
        2) sets the backend to the retrieved backend name
        """
        backend_name = os.environ.get(cls._ENV_DEFAULT_VAR, cls._default_backend)
        if backend_name not in cls.available_backend_names:
            msg = (f"{cls._ENV_DEFAULT_VAR} should be one of {''.join(map(repr, cls.available_backend_names))}"
                   f", got {backend_name}. Defaulting to {cls._default_backend}'")
            warnings.warn(msg, UserWarning)
            backend_name = cls._default_backend

        cls._default_backend = backend_name
        cls.set_backend(backend_name)

    @classmethod
    def load_backend(cls, backend_name):
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
        if backend_name not in cls.available_backend_names:
            msg = f"Unknown backend name {backend_name!r}, known backends are {cls.available_backend_names}"
            raise ValueError(msg)
        if backend_name not in Backend._available_backends:
            importlib.import_module('tensorly.backend.{0}_backend'.format(backend_name))
        if backend_name in Backend._available_backends:
            backend = Backend._available_backends[backend_name]()
            # backend = getattr(module, )()
            cls._loaded_backends[backend_name] = backend
        
        return backend

    @classmethod
    def set_backend(cls, backend, local_threadsafe=False):
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
            if backend not in cls._loaded_backends:
                backend = cls.load_backend(backend)
            else:
                backend = cls._loaded_backends[backend]

        cls._THREAD_LOCAL_DATA.backend = backend
        if not local_threadsafe:
            cls._default_backend = backend.backend_name
            cls._backend = backend

    @classmethod
    def register_backend_method(cls, name, fun_or_attr):
        cls.current_backend().register_method(name, fun_or_attr)

    def __dir__(cls):
        additionals = ['dynamically_dispatched_class_attribute', 'backend_manager', 'BackendManager']
        return cls.get_backend_dir() + additionals

# Initialise the backend to the default one
BackendManager.initialize_backend()
BackendManager.use_dynamic_dispatch()

sys.modules[__name__].__class__ = BackendManager
