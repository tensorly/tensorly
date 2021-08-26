from ..tenalg_utils import _validate_contraction_modes
import tensorly as tl

def tensordot(tensor1, tensor2, modes, batched_modes=()):
    """Batched tensor contraction between two tensors on specified modes
    
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

    start = ord('a')
    order_t1 = tl.ndim(tensor1)
    all_modes1 = [chr(start+i) for i in range(order_t1)]
    all_modes2 = [chr(start+i+order_t1) for i in range(tl.ndim(tensor2))]

    for m1, m2 in zip(modes1+batch_modes1, modes2+batch_modes2):
        all_modes2[m2] = all_modes1[m1]
    
    remaining_modes1 = [j for i, j in enumerate(all_modes1) if i not in modes1]
    remaining_modes2 = [j for i, j in enumerate(all_modes2) if i not in modes2+batch_modes2]
    remaining_modes = remaining_modes1 + remaining_modes2
    to_str = lambda x : ''.join(x)
    equation = f'{to_str(all_modes1)},{to_str(all_modes2)}->{to_str(remaining_modes)}'
    
    return tl.einsum(equation, tensor1, tensor2)
