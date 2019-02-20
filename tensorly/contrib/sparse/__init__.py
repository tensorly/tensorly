from ...backend import set_backend, get_backend, override_module_dispatch

from .backend import (tensor, is_tensor, context, shape, ndim, to_numpy, copy,
                      concatenate, reshape, moveaxis, transpose,
                      arange, ones, zeros, zeros_like, eye,
                      clip, where, max, min, all, mean, sum,
                      prod, sign, abs, sqrt, norm, dot, kron,
                      kr, solve, qr, partial_svd)


import sys
from ...backend import _get_backend_method, _get_backend_dir
static_items = list(sys.modules[__name__].__dict__.keys())
def sparse_dir():
    return _get_backend_dir() + static_items

override_module_dispatch(__name__, _get_backend_method, sparse_dir)
