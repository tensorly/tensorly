from ...kruskal_tensor import kruskal_to_tensor,unfolding_dot_khatri_rao
from .core import wrap

kruskal_to_tensor = wrap(kruskal_to_tensor)
unfolding_dot_khatri_rao = wrap(unfolding_dot_khatri_rao)
