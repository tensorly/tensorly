""" 
The :mod:`tensorly.tenalg` module contains utilities for Tensor Algebra 
operations such as khatri-rao or kronecker product, n-mode product, etc.
""" 

import sys
import importlib
import threading
import os
import warnings

from ..backend import BackendManager, dynamically_dispatched_class_attribute
from .base_tenalg import TenalgBackend

class TenalgBackendManager(BackendManager):
    _functions = ['mode_dot', 'multi_mode_dot', 
                  'kronecker', 'khatri_rao',
                  'inner', 'outer', 'batched_outer',
                  'higher_order_moment',
                  '_tt_matrix_to_tensor',
                  'tensordot'
                 ]
    _attributes = []
    available_backend_names = ['core', 'einsum']
    _default_backend = 'core'
    _loaded_backends = dict()
    _backend = None
    _THREAD_LOCAL_DATA = threading.local()
    _ENV_DEFAULT_VAR = 'TENSORLY_TENALG_BACKEND'

    @classmethod
    def use_dynamic_dispatch(cls):
        # Define class methods and attributes that dynamically dispatch to the backend
        for name in cls._functions:
            try:
                delattr(cls, name)
            except AttributeError:
                pass
            setattr(cls, name, staticmethod(cls.dispatch_backend_method(name, getattr(cls.current_backend(), name))))
        for name in cls._attributes:
            try:
                delattr(cls, name)
            except AttributeError:
                pass
            setattr(cls, name, dynamically_dispatched_class_attribute(name))
    
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
        if backend_name not in TenalgBackend._available_tenalg_backends:
            importlib.import_module('tensorly.tenalg.{0}_tenalg'.format(backend_name))
        if backend_name in TenalgBackend._available_tenalg_backends:
            backend = TenalgBackend._available_tenalg_backends[backend_name]()
            # backend = getattr(module, )()
            cls._loaded_backends[backend_name] = backend
        
        return backend


# Initialise the backend to the default one
TenalgBackendManager.initialize_backend()
TenalgBackendManager.use_dynamic_dispatch()

sys.modules[__name__].__class__ = TenalgBackendManager
