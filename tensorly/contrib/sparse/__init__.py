from ...backend import set_backend, get_backend

# from .core import (tensor, is_tensor, context, shape, ndim, to_numpy,
#                    copy, concatenate, reshape, moveaxis, transpose, arange, ones,
#                    zeros, zeros_like, eye, clip, where, max, min, all, mean,
#                    sum, prod, sign, abs, sqrt, norm, dot, kron, kr, solve, qr,
#                    partial_svd, unfold, fold, tensor_to_vec, vec_to_tensor,
#                    partial_unfold, partial_fold, partial_tensor_to_vec,
#                    partial_vec_to_tensor, kruskal_to_tensor,
#                    kruskal_to_unfolded, kruskal_to_vec, tucker_to_tensor,
#                    tucker_to_unfolded, tucker_to_vec, mps_to_tensor,
#                    mps_to_unfolded, mps_to_vec)

import sys
if sys.version_info[0] >=3 and sys.version_info[1] >= 7:
    from .backend import __getattr__
    from .backend import __dir__ as backend_dir

    # We need to add the above methods to the backend dir
    static_items = list(sys.modules[__name__].__dict__.keys())
    def __dir__():
        return backend_dir() + static_items

# Python 3.6 or lower: we need to overwrite the class of the module...
else:
    from .backend import _wrap_module
    _wrap_module(__name__)
