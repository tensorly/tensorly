from ..decomposition import parafac
from ..tenalg import multi_mode_dot
from ..kruskal_tensor import kruskal_to_tensor
from ....kruskal_tensor import kruskal_norm
from .... import backend as tl
from scipy import sparse
from ..backend import tensorflow_backend
from ....backend import set_backend, norm

import pytest
if not tl.get_backend() == "numpy":
    pytest.skip("Tests for sparse only with numpy backend", allow_module_level=True)
pytest.importorskip("sparse")

import numpy as np
import tensorflow as tf

def test_sparse_parafac():
    """Test for sparse parafac"""
    # Make sure the algorithm stays sparse. This will run out of memory on
    # most machines if the algorithm densifies.
    random_state = 1234
    rank = 3
    factors = [sparse.random((2862, rank), random_state=random_state),
               sparse.random((14036, rank), random_state=random_state)]
    weights = np.ones(rank)
    tensor = kruskal_to_tensor((weights, factors))
    _ = parafac(tensor, rank=rank, init='random', 
                n_iter_max=1, random_state=random_state)


def generate_random_sp_tensor(dimensions, d=0.0001):
    nd = len(dimensions)
    num_items = min(1000000, int(np.prod(dimensions) * d))

    idxs2 = np.random.rand(num_items, nd)
    idxs3 = np.trunc(idxs2 * dimensions).astype(int)
    idxs4 = [tuple(i) for i in idxs3]

    indices = list(set(idxs4))
    values = np.random.rand(len(indices))
    indices.sort()
    st = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dimensions)
    return st


def inner_product(kt, st):
    s = 0.0
    weights, factors = kt
    idxs = st.indices.numpy()
    vals = st.values.numpy()
    for i, index in enumerate(idxs):
        st_val = vals[i]
        kt_val = weights
        for fac_no, dim in enumerate(index):
            kt_val = np.multiply(factors[fac_no][dim], kt_val)
        s += (sum(kt_val) * st_val)
    return s


def fit(kt, st):
    normX = norm(st)
    normP = kruskal_norm(kt)
    ip = inner_product(kt, st)
    return 1 - ((normX**2 + normP**2 - 2*ip)**0.5)/normX


def test_tf_sparse_cpd():
    set_backend('tensorflow.sparse')

    print("generating tensor")
    shape = (100, 100, 1000)
    density = 0.001
    rank = 20

    st = generate_random_sp_tensor(shape, d=density)
    print("performing decomposition")
    cpd = parafac(st, rank, n_iter_max=50, verbose=True)
    print("testing fit")
    fit_st_rebuilt = fit(cpd, st)

    result_text = '''
    +--------------------------------------------
    | shape: {}
    | rank: {}
    | iterations: 100
    | density: {}
    | actual density: {}
    | number of non-zeros: {}
    |--------------------------------------------
    | fit: {}
    +--------------------------------------------
    '''.format(shape, rank, density, (st.values.shape[0] / np.prod(shape)), st.values.shape[0], fit_st_rebuilt)

    print(result_text)

