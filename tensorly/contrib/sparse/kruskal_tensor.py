from ...kruskal_tensor import kruskal_to_tensor,unfolding_dot_khatri_rao
from .core import wrap

from numpy import zeros
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
