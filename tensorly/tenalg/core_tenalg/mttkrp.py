from .n_mode_product import multi_mode_dot
from ... import backend as T

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

    Notes
    -----
    This is a variant of::

        unfolded = unfold(tensor, mode)
        kr_factors = khatri_rao(factors, skip_matrix=mode)
        mttkrp2 = tl.dot(unfolded, kr_factors)

    Multiplying with the Khatri-Rao product is equivalent to multiplying,
    for each rank, with the kronecker product of each factor.
    In code::

        mttkrp_parts = []
        for r in range(rank):
            component = tl.tenalg.multi_mode_dot(tensor, [f[:, r] for f in factors], skip=mode)
            mttkrp_parts.append(component)
        mttkrp = tl.stack(mttkrp_parts, axis=1)
        return mttkrp

    This can be done by taking n-mode-product with the full factors
    (faster but more memory consuming)::

        projected = multi_mode_dot(tensor, factors, skip=mode, transpose=True)
        ndims = T.ndim(tensor)
        res = []
        for i in range(factors[0].shape[1]):
            index = tuple([slice(None) if k == mode  else i for k in range(ndims)])
            res.append(projected[index])
        return T.stack(res, axis=-1)


    The same idea could be expressed using einsum::

        ndims = tl.ndim(tensor)
        tensor_idx = ''.join(chr(ord('a') + i) for i in range(ndims))
        rank = chr(ord('a') + ndims + 1)
        op = tensor_idx
        for i in range(ndims):
            if i != mode:
                op += ',' + ''.join([tensor_idx[i], rank])
            else:
                result = ''.join([tensor_idx[i], rank])
        op += '->' + result
        factors = [f for (i, f) in enumerate(factors) if i != mode]
        return tl_einsum(op, tensor, *factors)
    """
    mttkrp_parts = []
    weights, factors = cp_tensor
    rank = T.shape(factors[0])[1]
    for r in range(rank):
        component = multi_mode_dot(
            tensor, [T.conj(f[:, r]) for f in factors], skip=mode
        )
        mttkrp_parts.append(component)

    if weights is None:
        return T.stack(mttkrp_parts, axis=1)
    else:
        return T.stack(mttkrp_parts, axis=1) * T.reshape(weights, (1, -1))
