import tensorly as tl
from . import batched_outer


def higher_order_moment(tensor, order):
    """Computes the Higher-Order Momemt
    
    Parameters
    ----------
    tensor : 2D-tensor -- or ND-tensor
        matrix of size (n_samples, n_features)
        or tensor of size(n_samples, D1, ..., DN)
        
    order : int
        order of the higher-order moment to compute
        
    Returns
    -------
    tensor : moment
        if tensor is a matrix of size (n_samples, n_features), 
        tensor of size (n_features, )*order
    """    
    moment = tensor
    for _ in range(order - 1):
        moment = batched_outer(moment, tensor)
        
    return tl.mean(moment, axis=0)



