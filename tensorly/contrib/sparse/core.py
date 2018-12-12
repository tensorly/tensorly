import functools

from .backend import using_sparse_backend
from ... import backend, base, kruskal_tensor, tucker_tensor, mps_tensor


def wrap(func):
    @functools.wraps(func, assigned=('__name__', '__qualname__',
                                     '__doc__', '__annotations__'))
    def inner(*args, **kwargs):
        with using_sparse_backend():
            return func(*args, **kwargs)

    return inner


tensor = wrap(backend.tensor)
is_tensor = wrap(backend.is_tensor)
context = wrap(backend.context)
shape = wrap(backend.shape)
ndim = wrap(backend.ndim)
to_numpy = wrap(backend.to_numpy)
copy = wrap(backend.copy)
concatenate = wrap(backend.concatenate)
reshape = wrap(backend.reshape)
moveaxis = wrap(backend.moveaxis)
transpose = wrap(backend.transpose)
arange = wrap(backend.arange)
ones = wrap(backend.ones)
zeros = wrap(backend.zeros)
zeros_like = wrap(backend.zeros_like)
eye = wrap(backend.eye,)
clip = wrap(backend.clip)
where = wrap(backend.where)
max = wrap(backend.max)
min = wrap(backend.min)
all = wrap(backend.all)
mean = wrap(backend.mean)
sum = wrap(backend.sum)
prod = wrap(backend.prod)
sign = wrap(backend.sign)
abs = wrap(backend.abs)
sqrt = wrap(backend.sqrt)
norm = wrap(backend.norm)
dot = wrap(backend.dot)
kron = wrap(backend.kron)
kr = wrap(backend.kr)
solve = wrap(backend.solve)
qr = wrap(backend.qr)
partial_svd = wrap(backend.partial_svd)
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
