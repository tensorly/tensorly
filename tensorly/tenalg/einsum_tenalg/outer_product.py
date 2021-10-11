from ._batched_tensordot import tensordot

# Author: Jean Kossaifi

# License: BSD 3 clause

def outer(tensors):
    """Returns the outer product of tensors

    Parameters
    ----------
    tensors : tensor list

    Returns
    -------
    outer (tensor) product of the tensors
    """
    for i, tensor in enumerate(tensors):
        if i:
            res = tensordot(res, tensor, modes=(), batched_modes=())
        else:
            res = tensor
    return res

def batched_outer(tensors):
    """Returns the outer product of tensors

    Parameters
    ----------
    tensors : tensor list
        list of tensors of shape (batch-size, I_1, ..., I_N)

    Returns
    -------
    batched outer (tensor) product of the tensors
    """
    for i, tensor in enumerate(tensors):
        if i:
            res = tensordot(res, tensor, modes=(), batched_modes=(0))
        else:
            res = tensor
    return res
