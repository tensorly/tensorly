from ...tenalg import (mode_dot, multi_mode_dot, 
                       kronecker, khatri_rao, inner,
                       unfolding_dot_khatri_rao)
from .core import wrap


mode_dot = wrap(mode_dot)
multi_mode_dot = wrap(multi_mode_dot)
kronecker = wrap(kronecker)
khatri_rao = wrap(khatri_rao)
inner = wrap(inner)
unfolding_dot_khatri_rao = wrap(unfolding_dot_khatri_rao)
