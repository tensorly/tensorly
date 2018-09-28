import os
import sys
import types
import warnings
from functools import wraps
from threading import local

_STATE = local()
_STATE.backend = 'numpy'
_BACKENDS = ('numpy', 'mxnet', 'pytorch', 'tensorflow', 'cupy')


class set_backend(object):
    """Sets the backend for TensorLy.

    Can be used to set the backend globally, or locally as a contextmanager.

    Parameters
    ----------
    backend : {'numpy', 'mxnet', 'pytorch', 'tensorflow', 'cupy'}
        The backend to use. Default is 'numpy'.

    Examples
    --------
    Set the backend to numpy globally:

    >>> import tensorly as tl
    >>> tl.set_backend('numpy')

    Set the backend to use pytorch inside a context:

    >>> with tl.set_backend('pytorch'):  # doctest: +SKIP
    ...     pass
    """
    def __init__(self, backend):
        # load the backend
        if backend == 'numpy':
            from .backend import numpy_backend  # noqa
        elif backend == 'mxnet':
            from .backend import mxnet_backend  # noqa
        elif backend == 'pytorch':
            from .backend import pytorch_backend  # noqa
        elif backend == 'tensorflow':
            from .backend import tensorflow_backend  # noqa
        elif backend == 'cupy':
            from .backend import cupy_backend  # noqa
        else:
            raise ValueError("backend should be one of {%s}, got %r"
                             % (', '.join(map(repr, _BACKENDS)), backend))

        # Set the backend
        self._old_backend = _STATE.backend
        self._new_backend = _STATE.backend = backend

    def __repr__(self):
        return 'tensorly.set_backend(%r)' % self._new_backend

    def __enter__(self):
        return None

    def __exit__(self, *args):
        _STATE.backend = self._old_backend


def get_backend():
    """Returns the backend currently used

    Returns
    -------
    backend_used : str
        The backend currently in use
    """
    return _STATE.backend


class Registry(object):
    """A registry of methods and attributes for a backend to implement"""

    def __init__(self):
        self._methods = set()
        self._attributes = set()
        self._backends = {}

    def add_method(self, method):
        """Register a backend-dispatched method"""
        name = method.__name__
        self._methods.add(name)

        @wraps(method)
        def inner(*args, **kwargs):
            backend = self._backends[_STATE.backend]
            if name in backend.dispatch:
                return backend.dispatch[name](*args, **kwargs)
            raise NotImplementedError("Backend %r doesn't implement "
                                      "%r" % (_STATE.backend, name))

        return inner

    def add_attribute(self, name):
        """Register a backend-dispatched attribute"""
        self._attributes.add(name)

    def validate(self, name, obj):
        """Check that the object is valid for the given name"""
        if name in self._methods:
            # TODO: validate method signature
            pass
        if name not in self._methods and name not in self._attributes:
            raise ValueError("Unknown backend method/attribute %r" % name)

    def new_backend(self, name):
        if name in self._backends:
            raise ValueError("backend %r already exists" % name)

        backend = self._backends[name] = Backend(self, name)
        return backend

    def wrap_module(self, module_name):
        """Wrap a module to dynamically dispatch attributes to the registry.

        Intended use is

        >>> registry.wrap_module(__name__)
        """
        registry = self

        class RegistryModuleDispatch(types.ModuleType):
            def __getattr__(self, key):
                if key in registry._attributes:
                    backend = registry._backends[_STATE.backend]
                    if key in backend.dispatch:
                        return backend.dispatch[key]
                raise AttributeError("module %r has no attribute %r"
                                     % (self.__name__, key))

            def __dir__(self):
                out = set(super(RegistryModuleDispatch, self).__dir__())
                out.update(registry._attributes)
                return list(out)

        sys.modules[module_name].__class__ = RegistryModuleDispatch


class Backend(object):
    def __init__(self, registry, name):
        self.registry = registry
        self.name = name
        self.dispatch = {}

    def register(self, obj, name=None):
        """Register a method or object with the backend.

        Parameters
        ----------
        obj : object
            The object to register.
        name : str, optional
            The name to register the object with. If not provided, will be
            inferred from the object's name.
        """
        if name is None:
            name = obj.__name__
        if name in self.dispatch:
            raise ValueError("%r is already registered with "
                             "backend %r" % (name, self.name))
        self.registry.validate(name, obj)
        self.dispatch[name] = obj
        return obj


def _initialize_backend():
    default = 'numpy'
    backend = os.environ.get('TENSORLY_BACKEND', default)

    try:
        set_backend(backend)
    except ValueError:
        msg = ("TENSORLY_BACKEND should be one of {%s}, got %r. Defaulting to "
               "'%r'") % (', '.join(map(repr, _BACKENDS)), backend, default)
        warnings.warn(msg, UserWarning)
        set_backend(default)


_initialize_backend()
