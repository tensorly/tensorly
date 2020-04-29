from ..decomposition import parafac
from ..tenalg import multi_mode_dot
from ..kruskal_tensor import kruskal_to_tensor
from .... import backend as tl 

import pytest
if not tl.get_backend() == "numpy":
    pytest.skip("Tests for sparse only with numpy backend", allow_module_level=True)
pytest.importorskip("sparse")

import sparse
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

# generate random factors to make a kruskal tensor
def generate_random_factors(dimensions, rank, d = 0.1):
    factors = [sparse.random(dim,rank,density=d).A for dim in dimensions]
    return factors

# given a kruskal tensor, expands the kruskal tensor to a tensorflow.sparse.SparseTensor
def expand_random_factors(factors, dim_no, cur_idx, all_vals, rank):
    #this method just writes to all values, so all values needs to be saved somewhere
    if dim_no == len(factors):
        value = np.ones(rank)
        for i in range(dim_no):
            ci = cur_idx[i]
            f = factors[i][ci]
            for j,v in enumerate(value):
                value[j] = v * f[j]
        s = 0.0
        for val in value:
            s += val
        v = s * (3.16**dim_no)
        if(v != 0.0):
            t = np.ones(dim_no, dtype=np.int64)
            t *= cur_idx
            all_vals.append((t,v))
    else:
        cur_fact = factors[dim_no]
        for i in range(len(cur_fact)):
            cur_idx[dim_no] = i
            expand_random_factors(factors, dim_no + 1, cur_idx, all_vals, rank)

#creates a sparse tensor of rank 'rank'
def generate_decomposable_sp_tensor(dimensions, rank, d=0.00001):
    nd = len(dimensions)
    factor_d = (d / rank) ** (1 / nd)

    factors = generate_random_factors(dimensions, rank, factor_d)
    cur_idx = np.zeros(len(dimensions), dtype="int64")
    all_values = [(cur_idx, 0.0)]  # list(Tuple(array(int64, 1d, C), float64)))
    expand_random_factors(factors, 0, cur_idx, all_values, rank)
    all_values = all_values[1:]
    indices = [a[0] for a in all_values]
    values = [a[1] for a in all_values]
    shape = dimensions
    #     print(indices)
    #     print(values)
    st = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    return st


def tensor_norm(st):
    return (sum([x ** 2 for x in st.values.numpy()]) ** 0.5)


def diff(spt1, spt2):
    idx1 = [tuple(s) for s in spt1.indices.numpy()]
    idx2 = [tuple(s) for s in spt2.indices.numpy()]
    val1 = spt1.values.numpy()
    val2 = spt2.values.numpy()
    s1 = [(idx1[i], val1[i]) for i in range(len(idx1))]
    s2 = [(idx2[i], val2[i]) for i in range(len(idx2))]
    s1.sort()
    s2.sort()
    i1 = 0
    i2 = 0
    l1 = len(s1)
    l2 = len(s2)

    sum_sq = 0
    while (i1 < l1 and i2 < l2):
        p1 = s1[i1]
        p2 = s2[i2]
        if p1[0] == p2[0]:
            sum_sq += (p1[1] - p2[1]) ** 2
            i1 += 1
            i2 += 1
        elif p1[0] < p2[0]:
            sum_sq += p1[1] ** 2
            i1 += 1
        else:
            sum_sq += p2[1] ** 2
            i2 += 1
    if (i1 == l1):
        while (i2 < l2):
            p2 = s2[i2]
            sum_sq += p2[1] ** 2
            i2 += 1
    else:
        while (i1 < l1):
            p1 = s1[i1]
            sum_sq += p1[1] ** 2
            i1 += 1

    return sum_sq ** 0.5


def fit(spt1, spt2):
    return 1 - (diff(spt1, spt2) / tensor_norm(spt1))


def expand(factors, weights, dim_no, cur_idx, all_vals, rank):
    if dim_no == len(factors):
        #         print(cur_idx)
        value = np.ones(rank)
        for j, w in enumerate(weights):
            value[j] = w
        for i in range(dim_no):
            ci = cur_idx[i]
            f = factors[i][ci]
            for j, v in enumerate(value):
                value[j] = v * f[j]
        s = 0.0
        for val in value:
            s += val
        if (s != 0.0):
            t = np.ones(dim_no, dtype=np.int64)
            for k in range(dim_no):
                t[k] = cur_idx[k]
            all_vals.append((t, s))
    else:
        cur_fact = factors[dim_no]
        for i in range(len(cur_fact)):
            cur_idx[dim_no] = i
            expand(factors, weights, dim_no + 1, cur_idx, all_vals, rank)


def rebuild(kruskal_tensor, dimensions, rank):
    factors = kruskal_tensor[1]
    weights = kruskal_tensor[0][0]
    cur_idx = np.zeros(len(dimensions), dtype="int64")
    av = [(cur_idx, 0.0)]  # list(Tuple(array(int64, 1d, C), float64)))

    expand(factors, weights, 0, cur_idx, av, rank)

    av = av[1:]
    indexes = [a[0] for a in av]
    vals = [a[1] for a in av]
    st = tf.sparse.SparseTensor(indices=indexes, values=vals, dense_shape=dimensions)
    return st

def test_tf_sparse_cpd():
    st = generate_decomposable_sp_tensor((500,1000,500), 20, d=0.0001)
    cpd = cp_als(st, 20, n_iter_max=50)
    rebuilt = rebuild(cpd, shape, rank)
    fit_st_rebuilt = fit(st, rebuilt)

    result_text = '''
    +--------------------------------------------
    | shape: (500,1000,500)
    | rank: 20
    | iterations: 50
    | density: 0.0001
    | actual density: {}
    | number of non-zeros: {}
    |--------------------------------------------
    | fit: {}
    +--------------------------------------------
    '''.format((st.values.shape[0] / (500*1000*500)),
               st.values.shape[0], fit_st_rebuilt)

    print(result_text)
