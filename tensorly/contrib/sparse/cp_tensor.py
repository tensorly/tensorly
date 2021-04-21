from ...cp_tensor import cp_to_tensor,unfolding_dot_khatri_rao
from .core import wrap
from ...utils import DefineDeprecated

cp_to_tensor = wrap(cp_to_tensor)

kruskal_to_tensor = DefineDeprecated(deprecated_name='kruskal_to_tensor', use_instead=cp_to_tensor)

unfolding_dot_khatri_rao = wrap(unfolding_dot_khatri_rao)
