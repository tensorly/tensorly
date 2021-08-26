import tensorly as tl

# TODO : add batched_modes as in batched_tensor_dot? 
def batched_outer(tensors):
    """Returns a generalized outer product of the two tensors
    
    Parameters 
    ----------
    tensor1 : tensor
        of shape (n_samples, J1, ..., JN)
    tensor2 : tensor
        of shape (n_samples, K1, ..., KM)
        
    Returns
    -------
    outer product of tensor1 and tensor2 
        of shape (n_samples, J1, ..., JN, K1, ..., KM)
    """
    for i, tensor in enumerate(tensors):
        if i:

            shape = tl.shape(tensor)
            size = len(shape) - 1
            
            n_samples = shape[0]
            
            if n_samples != shape_res[0]:
                raise ValueError(f'Tensor {i} has a batch-size of {n_samples} but those before had a batch-size of {shape_res[0]}, '
                                'all tensors should have the same batch-size.')
            
            shape_1 = shape_res + (1, )*size
            shape_2 = (n_samples, ) + (1, )*size_res + shape[1:]

            res = tl.reshape(res, shape_1) * tl.reshape(tensor, shape_2)
        else:
            res = tensor

        shape_res = tl.shape(res)
        size_res = len(shape_res) - 1

    return res


def outer(tensors):
    """Returns a generalized outer product of the two tensors
    
    Parameters 
    ----------
    tensor1 : tensor
        of shape (J1, ..., JN)
    tensor2 : tensor
        of shape (K1, ..., KM)
        
    Returns
    -------
    outer product of tensor1 and tensor2 
        of shape (J1, ..., JN, K1, ..., KM)
    """
    for i, tensor in enumerate(tensors):
        if i:
            shape = tl.shape(tensor)
            s1 = len(shape)

            shape_1 = shape_res + (1, )*s1
            shape_2 = (1, )*sres + shape

            res = tl.reshape(res, shape_1) * tl.reshape(tensor, shape_2)
        else:
            res = tensor

        shape_res = tl.shape(res)
        sres = len(shape_res)
    
    return res