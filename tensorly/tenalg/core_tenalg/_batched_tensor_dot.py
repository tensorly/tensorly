from ..tenalg_utils import _validate_contraction_modes
import tensorly as tl
from math import prod


def batched_tensor_dot(tensor1, tensor2, modes, batched_modes=()):
    """Tensor contraction between two tensors on specified modes
    
    Parameters
    ----------
    tensor1 : tl.tensor
    tensor2 : tl.tensor
    modes : int list or int
        modes on which to contract tensor1 and tensor2
    batched_modes : int or tuple[int]

    Returns
    -------
    contraction : tensor1 contracted with tensor2 on the specified modes
    """
    modes1, modes2 = _validate_contraction_modes(tensor1.shape, tensor2.shape, modes)
    batch_modes1, batch_modes2 = _validate_contraction_modes(tensor1.shape, tensor2.shape, batched_modes, batched_modes=True)

    contraction_shape = [s for (i, s) in enumerate(tl.shape(tensor1)) if i in modes1]
    contraction_dim = prod(contraction_shape)
    
    n_free = tl.ndim(tensor1) - len(modes1) - len(batch_modes1)
    # We will reorganize tensor1 by just moving the contraction modes to the end
    modes_begin1 = []
    modes_end1 = []
    shape_begin2 = []
    last_mode_is_batched = False
    last_mode = tl.ndim(tensor1) - 1
    for i, s in enumerate(tl.shape(tensor1)):
        if i in batch_modes1:
            modes_begin1.append(i)
            shape_begin2.append(s)
            last_mode_is_batched = True
        elif i in modes1:
            modes_end1.append(i)
        else:
            modes_begin1.append(i)
            if i != last_mode or n_free:
                shape_begin2.append(1)
            last_mode_is_batched = False
    tensor1 = tl.transpose(tensor1, modes_begin1+modes_end1)

    n_modes_1 = tl.ndim(tensor1) - len(modes1)
    shape = list(tl.shape(tensor1))[:n_modes_1]
    if last_mode_is_batched:
        shape += [1]
    elif shape_begin2:
        shape_begin2.pop(-1)

    tensor1 = tl.reshape(tensor1, (*shape, contraction_dim))
    
    # these are neither batch-size nor contraction modes: put them last
    new_modes2 = [i for i in range(tensor2.ndim) if i not in batch_modes2+modes2]
    new_shape2 = [tl.shape(tensor2)[i] for i in new_modes2]
    tensor2 = tl.transpose(tensor2, batch_modes2+modes2+new_modes2)

    if not new_modes2:
        squeeze_last = True
    else:
        squeeze_last = False
    tensor2 = tl.reshape(tensor2, (*shape_begin2, contraction_dim, -1))
    
    res = tl.matmul(tensor1, tensor2)
    
    out_shape = list(tl.shape(res))
    if squeeze_last:
        out_shape = out_shape[:-1]
    else:
        out_shape = out_shape[:-1] + new_shape2
    if last_mode_is_batched:
        out_shape.pop(n_modes_1)
    
    return tl.reshape(res, out_shape)

