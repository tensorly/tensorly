from .n_mode_product import mode_dot, multi_mode_dot
from ._kronecker import kronecker
from ._khatri_rao import khatri_rao
from .generalised_inner_product import inner
from .outer_product import outer, batched_outer
from .moments import higher_order_moment
from ._tt_matrix import tt_matrix_to_tensor
from ._batched_tensordot import tensordot

from ..base_tenalg import TenalgBackend

class CoreTenalgBackend(TenalgBackend, backend_name='core'):
    pass

CoreTenalgBackend.register_method('mode_dot', mode_dot)
CoreTenalgBackend.register_method('multi_mode_dot', multi_mode_dot)
CoreTenalgBackend.register_method('kronecker', kronecker)
CoreTenalgBackend.register_method('khatri_rao', khatri_rao)
CoreTenalgBackend.register_method('inner', inner)
CoreTenalgBackend.register_method('outer', outer)
CoreTenalgBackend.register_method('batched_outer', batched_outer)
CoreTenalgBackend.register_method('higher_order_moment', higher_order_moment)
CoreTenalgBackend.register_method('_tt_matrix_to_tensor', tt_matrix_to_tensor)
CoreTenalgBackend.register_method('tensordot', tensordot)