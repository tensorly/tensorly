__version__ = '0.6.1'

import sys

from .base import unfold, fold
from .base import tensor_to_vec, vec_to_tensor
from .base import partial_unfold, partial_fold
from .base import partial_tensor_to_vec, partial_vec_to_tensor

from .cp_tensor import (cp_to_tensor, cp_to_unfolded,
                        cp_to_vec, unfolding_dot_khatri_rao, 
                        cp_norm, cp_mode_dot, cp_normalize,
                        validate_cp_rank)
from .tucker_tensor import (tucker_to_tensor, tucker_to_unfolded,
                            tucker_to_vec, tucker_mode_dot,
                            validate_tucker_rank)
from .tt_tensor import tt_to_tensor, tt_to_unfolded, tt_to_vec, validate_tt_rank
from .tt_matrix import (tt_matrix_to_tensor, tt_matrix_to_tensor, validate_tt_matrix_rank,
                        tt_matrix_to_unfolded, tt_matrix_to_vec, tt_matrix_to_matrix)

from .backend import (set_backend, get_backend,
                      backend_context, _get_backend_dir,
                      _get_backend_method, override_module_dispatch)

from .backend import (context, tensor, is_tensor, shape, ndim, to_numpy, copy,
                      concatenate, reshape, transpose, moveaxis, arange, ones,
                      zeros, zeros_like, eye, where, clip, max, min, argmax,
                      argmin, all, mean, sum, prod, sign, abs, sqrt, norm, dot,
                      kron, solve, qr, kr, partial_svd, stack, eps, finfo,
                      index_update, check_random_state, randomized_svd,
                      randomized_range_finder, log2, sin, cos)


# Deprecated
from .cp_tensor import kruskal_to_tensor, kruskal_to_unfolded, kruskal_to_vec


# Add Backend functions, dynamically dispatched
def full_dir():
    """Returns the module's __dir__, including the local variables
        and augmenting it with the dynamically dispatched variables from backend.
    """
    static_items = list(sys.modules[__name__].__dict__.keys())
    return _get_backend_dir() + static_items

override_module_dispatch(__name__, _get_backend_method, full_dir)
del override_module_dispatch, full_dir, _get_backend_method
