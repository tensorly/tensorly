import sys
import importlib
import os

# Set the default backend
default_backend = 'mxnet'
try:
    if _BACKEND is None:
        _BACKEND = os.environ.get('TENSORLY_BACKEND', default_backend)
except NameError:
    _BACKEND = os.environ.get('TENSORLY_BACKEND', default_backend)

def set_backend(backend_name):
    global _BACKEND
    _BACKEND = backend_name

    # reloads tensorly.backend
    importlib.reload(backend)

    # reload from .backend import * (e.g. tensorly.tensor)
    globals().update(
            {fun: getattr(backend, fun) for n in backend.__all__} if hasattr(backend, '__all__') 
            else 
            {k: v for (k, v) in backend.__dict__.items() if not k.startswith('_')
            })


from .backend import *
from .base import unfold, fold
from .base import tensor_to_vec, vec_to_tensor                                                                                                           
from .base import partial_unfold, partial_fold
from .base import partial_tensor_to_vec, partial_vec_to_tensor

from .kruskal_tensor import kruskal_to_tensor, kruskal_to_unfolded, kruskal_to_vec
from .tucker_tensor import tucker_to_tensor, tucker_to_unfolded, tucker_to_vec


__version__ = '0.2.0'



