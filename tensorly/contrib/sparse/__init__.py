from ...backend import set_backend, get_backend
from ... import backend, base, cp_tensor, tucker_tensor, tt_tensor

from .backend import (tensor, is_tensor, context, shape, ndim, to_numpy, copy,
                      concatenate, reshape, moveaxis, transpose,
                      arange, ones, zeros, zeros_like, eye,
                      clip, where, max, min, all, mean, sum,
                      prod, sign, abs, sqrt, norm, dot, kron,
                      kr, solve, qr, partial_svd)

from .core import wrap

import sys
# from ...backend import _get_backend_method, _get_backend_dir
# from ...backend import backend
from ... import backend
static_items = list(sys.modules[__name__].__dict__.keys())

def __dir__():
    return backend.get_backend_dir() + static_items
    # return _get_backend_dir() + static_items

__getattr__ = backend.__getattribute__
# override_module_dispatch(__name__, backend_manager.__getattribute__, sparse_dir)
# override_module_dispatch(__name__, _get_backend_method, sparse_dir)

unfold = wrap(base.unfold)
fold = wrap(base.fold)
tensor_to_vec = wrap(base.tensor_to_vec)
vec_to_tensor = wrap(base.vec_to_tensor)
partial_unfold = wrap(base.partial_unfold)
partial_fold = wrap(base.partial_fold)
partial_tensor_to_vec = wrap(base.partial_tensor_to_vec)
partial_vec_to_tensor = wrap(base.partial_vec_to_tensor)
cp_to_tensor = wrap(cp_tensor.cp_to_tensor)
cp_to_unfolded = wrap(cp_tensor.cp_to_unfolded)
cp_to_vec = wrap(cp_tensor.cp_to_vec)
tucker_to_tensor = wrap(tucker_tensor.tucker_to_tensor)
tucker_to_unfolded = wrap(tucker_tensor.tucker_to_unfolded)
tucker_to_vec = wrap(tucker_tensor.tucker_to_vec)
tt_to_tensor = wrap(tt_tensor.tt_to_tensor)
tt_to_unfolded = wrap(tt_tensor.tt_to_unfolded)
tt_to_vec = wrap(tt_tensor.tt_to_vec)

# Deprecated
kruskal_to_tensor = wrap(cp_tensor.kruskal_to_tensor)
kruskal_to_unfolded = wrap(cp_tensor.kruskal_to_unfolded)
kruskal_to_vec = wrap(cp_tensor.kruskal_to_vec)
