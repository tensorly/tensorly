from ...backend import set_backend, get_backend, override_module_dispatch

from .core import (tensor, is_tensor, context, shape, ndim, to_numpy,
                   copy, concatenate, reshape, moveaxis, transpose, arange, ones,
                   zeros, zeros_like, eye, clip, where, max, min, all, mean,
                   sum, prod, sign, abs, sqrt, norm, dot, kron, kr, solve, qr,
                   partial_svd, unfold, fold, tensor_to_vec, vec_to_tensor,
                   partial_unfold, partial_fold, partial_tensor_to_vec,
                   partial_vec_to_tensor, kruskal_to_tensor,
                   kruskal_to_unfolded, kruskal_to_vec, tucker_to_tensor,
                   tucker_to_unfolded, tucker_to_vec, mps_to_tensor,
                   mps_to_unfolded, mps_to_vec)

import sys
from .backend import get_backend_method, get_backend_dir
static_items = list(sys.modules[__name__].__dict__.keys())
def sparse_dir():
    return get_backend_dir() + static_items

override_module_dispatch(__name__, get_backend_method, sparse_dir)
