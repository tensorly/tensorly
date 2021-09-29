import importlib
from contextlib import contextmanager
import functools

from ....backend import backend_context, get_backend, override_module_dispatch
from .... import backend, base, cp_tensor, tucker_tensor, tt_tensor


_KNOWN_BACKENDS = {'numpy': 'NumpySparseBackend'}
# Mapping name: funs are stored here
_LOADED_BACKENDS = {}

dispatched_sparse = ['reshape', 'moveaxis', 'any', 'trace', 'shape', 'ndim',
                        'where', 'copy', 'transpose', 'arange', 'ones', 'zeros',
                        'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min', 'matmul',
                        'all', 'mean', 'sum', 'cumsum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
                        'argmax', 'stack', 'conj', 'diag', 'einsum', 'log2', 'dot', 'tensordot', 
                        'sin', 'cos', 'clip', 'kr', 'kron', 'partial_svd',
                        'solve', 'qr', 'svd', 'eigh', 'randn', 'check_random_state',
                        'index_update', 'context', 'tensor', 'norm', 'to_numpy', 'is_tensor'
                       ]


@contextmanager
def sparse_context():
    backend_name = get_backend()
    if backend_name not in _LOADED_BACKENDS:
        register_sparse_backend(backend_name)

    with backend_context(_LOADED_BACKENDS[backend_name]):
        yield

def register_sparse_backend(backend_name):
    """Registers a new backend by importing the corresponding module 
        and adding the correspond `Backend` class in Backend._LOADED_BACKEND
        under the key `backend_name`
    
    Parameters
    ----------
    backend_name : str, name of the backend to load
    
    Raises
    ------
    ValueError
        If `backend_name` does not correspond to one listed
            in `_KNOWN_BACKEND`
    """
    if backend_name in _KNOWN_BACKENDS:
        module = importlib.import_module('tensorly.contrib.sparse.backend.{0}_backend'.format(backend_name))
        backend = getattr(module, _KNOWN_BACKENDS[backend_name])()
        default_backend_mapping = {name:getattr(backend, name) for name in dispatched_sparse}
        _LOADED_BACKENDS[backend_name] = default_backend_mapping
    else:
        msg = "Unknown backend name {0!r}, known backends are [{1}]".format(
                backend_name, ', '.join(map(repr, _KNOWN_BACKENDS)))
        raise ValueError(msg)

def _get_backend_method(method_name):
    backend_name = get_backend()
    if backend_name not in _LOADED_BACKENDS:
        register_sparse_backend(backend_name)

    return getattr(_LOADED_BACKENDS[backend_name], method_name)

def _get_backend_dir():
    backend_name = get_backend()
    if backend_name not in _LOADED_BACKENDS:
        register_sparse_backend(backend_name)

    return [k for k in dir(_LOADED_BACKENDS[backend_name]) if not k.startswith('_')]

override_module_dispatch(__name__, _get_backend_method, _get_backend_dir)

def dispatch_sparse(func):
    # global dispatched_sparse
    # dispatched_sparse.append(func.__name__)
    @functools.wraps(func, assigned=('__name__', '__qualname__',
                                     '__doc__', '__annotations__'))
    def inner(*args, **kwargs):
        with sparse_context():
            return func(*args, **kwargs)

    return inner

tensor = dispatch_sparse(backend.tensor)
is_tensor = dispatch_sparse(backend.is_tensor)
context = dispatch_sparse(backend.context)
shape = dispatch_sparse(backend.shape)
ndim = dispatch_sparse(backend.ndim)
to_numpy = dispatch_sparse(backend.to_numpy)
copy = dispatch_sparse(backend.copy)
concatenate = dispatch_sparse(backend.concatenate)
reshape = dispatch_sparse(backend.reshape)
moveaxis = dispatch_sparse(backend.moveaxis)
transpose = dispatch_sparse(backend.transpose)
arange = dispatch_sparse(backend.arange)
ones = dispatch_sparse(backend.ones)
zeros = dispatch_sparse(backend.zeros)
zeros_like = dispatch_sparse(backend.zeros_like)
eye = dispatch_sparse(backend.eye,)
clip = dispatch_sparse(backend.clip)
where = dispatch_sparse(backend.where)
max = dispatch_sparse(backend.max)
min = dispatch_sparse(backend.min)
all = dispatch_sparse(backend.all)
mean = dispatch_sparse(backend.mean)
sum = dispatch_sparse(backend.sum)
prod = dispatch_sparse(backend.prod)
sign = dispatch_sparse(backend.sign)
abs = dispatch_sparse(backend.abs)
sqrt = dispatch_sparse(backend.sqrt)
norm = dispatch_sparse(backend.norm)
dot = dispatch_sparse(backend.dot)
kron = dispatch_sparse(backend.kron)
kr = dispatch_sparse(backend.kr)
solve = dispatch_sparse(backend.solve)
qr = dispatch_sparse(backend.qr)
partial_svd = dispatch_sparse(backend.partial_svd)
unfold = dispatch_sparse(base.unfold)
fold = dispatch_sparse(base.fold)
tensor_to_vec = dispatch_sparse(base.tensor_to_vec)
vec_to_tensor = dispatch_sparse(base.vec_to_tensor)
partial_unfold = dispatch_sparse(base.partial_unfold)
partial_fold = dispatch_sparse(base.partial_fold)
partial_tensor_to_vec = dispatch_sparse(base.partial_tensor_to_vec)
partial_vec_to_tensor = dispatch_sparse(base.partial_vec_to_tensor)
cp_to_tensor = dispatch_sparse(cp_tensor.cp_to_tensor)
cp_to_unfolded = dispatch_sparse(cp_tensor.cp_to_unfolded)
cp_to_vec = dispatch_sparse(cp_tensor.cp_to_vec)
tucker_to_tensor = dispatch_sparse(tucker_tensor.tucker_to_tensor)
tucker_to_unfolded = dispatch_sparse(tucker_tensor.tucker_to_unfolded)
tucker_to_vec = dispatch_sparse(tucker_tensor.tucker_to_vec)
tt_to_tensor = dispatch_sparse(tt_tensor.tt_to_tensor)
tt_to_unfolded = dispatch_sparse(tt_tensor.tt_to_unfolded)
tt_to_vec = dispatch_sparse(tt_tensor.tt_to_vec)