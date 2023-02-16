import tensorly as tl

# Author: Jean Kossaifi


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
    ndims = tl.ndim(tensor)

    if weights is None:
        weights = tl.ones(factors[0].shape[1], **tl.context(tensor))

    tensor_idx = "".join(chr(ord("a") + i) for i in range(ndims))
    rank = chr(ord("a") + ndims + 1)
    op = tensor_idx + "," + rank
    for i in range(ndims):
        if i != mode:
            op += "," + "".join([tensor_idx[i], rank])
        else:
            result = "".join([tensor_idx[i], rank])
    op += "->" + result
    factors = [tl.conj(f) for (i, f) in enumerate(factors) if i != mode]
    return tl.einsum(op, tensor, weights, *factors)
