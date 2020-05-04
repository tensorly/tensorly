from ...kruskal_tensor import kruskal_to_tensor, unfolding_dot_khatri_rao, kruskal_norm
from .core import wrap

from .backend import norm

from numpy import zeros, multiply
import tensorly as tl


kruskal_to_tensor = wrap(kruskal_to_tensor)
unfolding_dot_khatri_rao = wrap(unfolding_dot_khatri_rao)


def sparse_mttkrp(tensor, factors, n, rank, dims):
    values = tl.values(tensor)
    indices = tl.indices(tensor)
    output = zeros((dims[n], rank))

    for l in range(len(values)):
        cur_index = indices[l]
        prod = [values[l]] * rank  # makes the value into a row

        for mode, cv in enumerate(cur_index):  # does elementwise row multiplications
            if mode != n:
                for r in range(rank):
                    prod[r] *= factors[mode][cv][r]

        for r in range(rank):
            output[cur_index[n]][r] += prod[r]

    return output


def kruskal_sparse_inner_product(kt, st):
    s = 0.0
    weights, factors = kt
    idxs = st.indices.numpy()
    vals = st.values.numpy()
    for i, index in enumerate(idxs):
        st_val = vals[i]
        kt_val = weights
        for fac_no, dim in enumerate(index):
            kt_val = multiply(factors[fac_no][dim], kt_val)
        s += (sum(kt_val) * st_val)
    return s


def kruskal_sparse_fit(kt, st):
        normX = norm(st)
        normP = kruskal_norm(kt)
        ip = kruskal_sparse_inner_product(kt, st)
        return 1 - ((normX ** 2 + normP ** 2 - 2 * ip) ** 0.5) / normX