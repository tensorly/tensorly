from .n_mode_product import mode_dot, multi_mode_dot
from ._kronecker import kronecker
from ._khatri_rao import khatri_rao
from .generalised_inner_product import inner
from .outer_product import outer, batched_outer
from .moments import higher_order_moment
from ._tt_matrix import tt_matrix_to_tensor as _tt_matrix_to_tensor
from ._batched_tensordot import tensordot
