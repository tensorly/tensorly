__version__ = '0.4.5'
import sys

from .base import unfold, fold
from .base import tensor_to_vec, vec_to_tensor
from .base import partial_unfold, partial_fold
from .base import partial_tensor_to_vec, partial_vec_to_tensor

from .kruskal_tensor import (kruskal_to_tensor, kruskal_to_unfolded,
                             kruskal_to_vec, unfolding_dot_khatri_rao)
from .tucker_tensor import tucker_to_tensor, tucker_to_unfolded, tucker_to_vec
from .mps_tensor import mps_to_tensor, mps_to_unfolded, mps_to_vec

from .backend import (set_backend, get_backend,
                      backend_context, _get_backend_dir,
                      _get_backend_method, override_module_dispatch)

from .backend import (context, tensor, is_tensor, shape, ndim, to_numpy, copy,
                      concatenate, reshape, transpose, moveaxis, arange, ones,
                      zeros, zeros_like, eye, where, clip, max, min, argmax,
                      argmin, all, mean, sum, prod, sign, abs, sqrt, norm, dot,
                      kron, solve, qr, kr, partial_svd, stack)


def full_dir():
    static_items = list(sys.modules[__name__].__dict__.keys())
    return _get_backend_dir() + static_items

override_module_dispatch(__name__, _get_backend_method, full_dir)
del override_module_dispatch, full_dir, _get_backend_method
