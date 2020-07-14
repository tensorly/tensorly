from ..decomposition import parafac
from ..tenalg import multi_mode_dot
from ..kruskal_tensor import kruskal_to_tensor, kruskal_sparse_inner_product
from .... import backend as tl
from scipy import sparse
from ..kruskal_tensor import kruskal_sparse_fit
from ....backend import set_backend, get_backend
from ..backend import sparse_context

import numpy as np

import pytest
if not tl.get_backend() == "numpy":
    pytest.skip("Tests for sparse only with numpy backend", allow_module_level=True)
pytest.importorskip("sparse")



def test_sparse_parafac():
    """Test for sparse parafac"""
    # Make sure the algorithm stays sparse. This will run out of memory on
    # most machines if the algorithm densifies.
    random_state = 1234
    rank = 3
    factors = [sparse.random(2862, rank, random_state=random_state),
               sparse.random(14036, rank, random_state=random_state)]
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
    indices = np.asarray(indices)
    st = tl.tensor((indices, values, dimensions))
    return st


def test_tf_sparse_cpd():
    with sparse_context():
        if get_backend() == 'tensorflow':
            print("generating tensor")
            shape = (100, 100, 100)
            density = 0.001
            rank = 20

            st = generate_random_sp_tensor(shape, d=density)
            print("performing decomposition")
            cpd = parafac(st, rank, n_iter_max=50, verbose=True)
            print("testing fit")
            fit_st_rebuilt = kruskal_sparse_fit(cpd, st)

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


