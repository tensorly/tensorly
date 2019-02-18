import functools

from .backend import sparse_context
from ... import backend, base, kruskal_tensor, tucker_tensor, mps_tensor

def wrap(func):
    @functools.wraps(func, assigned=('__name__', '__qualname__',
                                     '__doc__', '__annotations__'))
    def inner(*args, **kwargs):
        with sparse_context():
            return func(*args, **kwargs)

    return inner

unfold = wrap(base.unfold)
fold = wrap(base.fold)
tensor_to_vec = wrap(base.tensor_to_vec)
vec_to_tensor = wrap(base.vec_to_tensor)
partial_unfold = wrap(base.partial_unfold)
partial_fold = wrap(base.partial_fold)
partial_tensor_to_vec = wrap(base.partial_tensor_to_vec)
partial_vec_to_tensor = wrap(base.partial_vec_to_tensor)
kruskal_to_tensor = wrap(kruskal_tensor.kruskal_to_tensor)
kruskal_to_unfolded = wrap(kruskal_tensor.kruskal_to_unfolded)
kruskal_to_vec = wrap(kruskal_tensor.kruskal_to_vec)
tucker_to_tensor = wrap(tucker_tensor.tucker_to_tensor)
tucker_to_unfolded = wrap(tucker_tensor.tucker_to_unfolded)
tucker_to_vec = wrap(tucker_tensor.tucker_to_vec)
mps_to_tensor = wrap(mps_tensor.mps_to_tensor)
mps_to_unfolded = wrap(mps_tensor.mps_to_unfolded)
mps_to_vec = wrap(mps_tensor.mps_to_vec)
