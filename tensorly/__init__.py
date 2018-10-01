__version__ = '0.4.3'

from .backend import (set_backend, get_backend,
                      tensor, is_tensor,
                      context, shape, ndim,
                      to_numpy, copy,
                      concatenate, reshape, moveaxis, transpose,
                      arange, ones, zeros, zeros_like, eye,
                      clip, where,
                      max, min, all, mean, sum, prod,
                      sign, abs, sqrt,
                      norm, dot, kron, kr, solve, qr, partial_svd)

from .base import unfold, fold
from .base import tensor_to_vec, vec_to_tensor
from .base import partial_unfold, partial_fold
from .base import partial_tensor_to_vec, partial_vec_to_tensor

from .kruskal_tensor import kruskal_to_tensor, kruskal_to_unfolded, kruskal_to_vec
from .tucker_tensor import tucker_to_tensor, tucker_to_unfolded, tucker_to_vec
from .mps_tensor import mps_to_tensor, mps_to_unfolded, mps_to_vec

from .backend.core import wrap_module
wrap_module(__name__)
del wrap_module
