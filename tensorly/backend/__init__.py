from .core import set_backend, get_backend
from .core import (context, tensor, is_tensor, shape, ndim, to_numpy, copy,
                   concatenate, reshape, transpose, moveaxis, arange, ones,
                   zeros, zeros_like, eye, where, clip, max, min, all, mean,
                   sum, prod, sign, abs, sqrt, norm, dot, kron, solve, qr, kr,
                   partial_svd)

from .core import wrap_module
wrap_module(__name__)
del wrap_module
