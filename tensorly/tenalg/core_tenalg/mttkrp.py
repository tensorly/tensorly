from .n_mode_product import multi_mode_dot
from ._khatri_rao import khatri_rao
from ... import backend as T
from ...base import unfold

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
    Default unfolding_dot_khatri_rao implementation.

    Implemented as the product between an unfolded tensor
    and a Khatri-Rao product explicitly formed. Due to matrix-matrix
    products being extremely efficient operations, this is a
    simple yet hard-to-beat implementation of MTTKRP.

    If working with sparse tensors, or when the CP-rank of the CP-tensor is comparable to, or larger than,
    the dimensions of the input tensor, however, this method requires a lot
    of memory, which can be harmful when dealing with large tensors. In this
    case, please use the memory-efficient version of MTTKRP.

    To use the slower memory efficient version, run

    >>> from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao_memory
    >>> tl.tenalg.register_backend_method("unfolding_dot_khatri_rao", unfolding_dot_khatri_rao_memory)
    >>> tl.tenalg.use_dynamic_dispatch()

    """
    weights, factors = cp_tensor
    kr_factors = khatri_rao(factors, weights=weights, skip_matrix=mode)
    mttkrp = T.dot(unfold(tensor, mode), T.conj(kr_factors))
    return mttkrp


def unfolding_dot_khatri_rao_memory(tensor, cp_tensor, mode):
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
    Implemented as a sequence of Tensor-times-vectors products between a tensor
    and a Khatri-Rao product. The Khatri-Rao product is never computed explicitly,
    rather each column in the Khatri-Rao product is contracted with the tensor. This
    operation is implemented in Python and without making of use of parallelism, and it
    is therefore in general slower than the naive MTTKRP product.
    When the CP-rank of the CP-tensor is comparable to, or larger than,
    the dimensions of the input tensor, this method however requires much less
    memory.

    This method can also be implemented by taking n-mode-product with the full factors
    (faster but more memory consuming)::

        projected = multi_mode_dot(tensor, factors, skip=mode, transpose=True)
        ndims = T.ndim(tensor)
        res = []
        for i in range(factors[0].shape[1]):
            index = tuple([slice(None) if k == mode  else i for k in range(ndims)])
            res.append(projected[index])
        return T.stack(res, axis=-1)
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
