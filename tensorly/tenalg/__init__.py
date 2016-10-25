"""
The :mod:`tensorly.tenalg` module contains utilities for Tensor Algebra
operations such as khatri-rao or kronecker product, n-mode product, fast
partial-SVD, higher-order moments, etc.
"""

from ._khatri_rao import khatri_rao
from ._kronecker import kronecker
from .n_mode_product import mode_dot, multi_mode_dot
from ._norm import norm
from ._partial_svd import partial_svd
from ._higher_order_moment import higher_order_moment
