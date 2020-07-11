import tensorly as tl

def batched_tensor_dot(tensor1, tensor2):
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
    shape_1 = tl.shape(tensor1)
    s1 = len(shape_1) - 1
    shape_2 = tl.shape(tensor2)
    s2 = len(shape_2) - 1
    
    n_samples = shape_2[0]
    
    if n_samples != shape_1[0]:
        raise ValueError(f'tensor1 has a batch-size of {s1[0]} but tensor2 has a batch-size of {n_samples}'
                         'tensor1 and tensor2 should have the same batch-size.')
    
    shape_1 = shape_1 + (1, )*s2
    shape_2 = (n_samples, ) + (1, )*s1 + shape_2[1:]
    
    return tl.reshape(tensor1, shape_1) * tl.reshape(tensor2, shape_2)


def tensor_dot(tensor1, tensor2):
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
    shape_1 = tl.shape(tensor1)
    s1 = len(shape_1)
    shape_2 = tl.shape(tensor2)
    s2 = len(shape_2)
    
    shape_1 = shape_1 + (1, )*s2
    shape_2 = (1, )*s1 + shape_2
    
    return tl.reshape(tensor1, shape_1) * tl.reshape(tensor2, shape_2)