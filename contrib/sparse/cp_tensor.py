from ...cp_tensor import cp_to_tensor
from ...tenalg import unfolding_dot_khatri_rao

from .core import wrap

cp_to_tensor = wrap(cp_to_tensor)


unfolding_dot_khatri_rao = wrap(unfolding_dot_khatri_rao)
