from .base import unfold, fold
from .base import tensor_to_vec, vec_to_tensor
from .base import partial_unfold, partial_fold
from .base import partial_tensor_to_vec, partial_vec_to_tensor
from .base import tensor_from_frontal_slices

from .kruskal import kruskal_to_tensor, kruskal_to_unfolded, kruskal_to_vec
from .tucker import tucker_to_tensor, tucker_to_unfolded, tucker_to_vec

__version__ = '0.1.2'
