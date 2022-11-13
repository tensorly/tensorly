import tensorly as tl
from .caching import einsum_path_cached

# Author: Jean Kossaifi


@einsum_path_cached
def unfolding_dot_khatri_rao_path(tensor, mode):
    ndims = tl.ndim(tensor)
    tensor_idx = "".join(chr(ord("a") + i) for i in range(ndims))
    rank = chr(ord("a") + ndims + 1)
    op = tensor_idx + "," + rank
    for i in range(ndims):
        if i != mode:
            op += "," + "".join([tensor_idx[i], rank])
        else:
            result = "".join([tensor_idx[i], rank])
    op += "->" + result
    return op


def unfolding_dot_khatri_rao(tensor, cp_tensor, mode):
    """mode-n unfolding times khatri-rao product of factors

    Parameters
    ----------
    tensor : tl.tensor
        tensor to unfold
    factors : tl.tensor list
        list of matrices of which to the khatri-rao product
    mode : int
        mode on which to unfold `tensor`

    Returns
    -------
    mttkrp
        dot(unfold(tensor, mode), khatri-rao(factors))
    """
    weights, factors = cp_tensor

    key = f"{tl.shape(tensor)},{mode}"
    equation = unfolding_dot_khatri_rao_path(key, tensor, mode)

    factors = [f for (i, f) in enumerate(factors) if i != mode]
    return tl.einsum(equation, tensor, weights, *factors)
