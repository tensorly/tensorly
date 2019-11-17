import tensorly as tl
from .base import unfold

import torch

def get_scale_denominator(dtype):
    """Returns a number of intervals into which we divide
    a range of valid values [min_value, max_malue] during quantization.
    
    Parameters
    ----------
    dtype : torch.dtype
        Type to which we cast during quantization.
    
    Returns
    -------
    int
        Number of quantization intervals.
    """
    
    if dtype == torch.qint8:
        scale_denom = 2**8 - 1  
    elif dtype == torch.qint32:
        scale_denom = 2**32 - 1
    else:
        raise TypeError("Can't perform quantization. Unknown quantization type: {}".format(dtype))
        return
    
    return scale_denom


def get_per_channel_stats(tensor, mode = 0):  
    """Returns max, min and mean along last dimension for mode-`mode` unfolding
    of `tensor` with modes starting at `0`.
    
    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
          Indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``.

    Returns
    -------
    tuple
        max, min, mean of unfolded_tensor along last dimension.
    """
    unfolded_tensor = unfold(tensor, mode = mode)
    
    tmax = unfolded_tensor.max(dim = -1)[0]
    tmin = unfolded_tensor.min(dim = -1)[0]
    tmean = unfolded_tensor.mean(dim = -1)
    
    return tmax, tmin, tmean


def get_scale_zeropoint(tensor,\
                        dtype = torch.qint8,\
                        qscheme = torch.per_tensor_affine,\
                        dim = None):
    """Returns scale and zero_point to apply in quantization formula.
    
    Parameters
    ----------
    tensor : Tensor
        Float tensor to quantize.
    dtype : ``torch.dtype``, default is ``torch.qint8``
        The desired data type of returned tensor.
        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``.
    qscheme : quantization scheme, default is ``torch.per_tensor_affine``
        Has to be one of: ``torch.per_tensor_affine``, ``torch.per_tensor_symmetric``, ``torch.per_channel_affine``, ``torch.per_channel_symmetric``
    dim : int or None, default is None
        If dim is not None, along the dimension `dim` the values in the `tensor` are scaled and offset by a different value (effectively the scale and offset become vectors).
        If dim is None, all values in the `tensor` are scaled and offset by the same value.
    
    Returns
    -------
    scale
        Scale to apply in quantization formula.
    zero_point
        Offset in integer value that maps to float zero.
    """
    
    scale_denom = get_scale_denominator(dtype)
  
    if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
        
        tmax, tmin, zero_point = get_per_channel_stats(tensor, mode = dim)
        zero_point = zero_point.int()

    elif qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
        tmax = tensor.max()
        tmin = tensor.min() 
        zero_point = tensor.mean().int()
        
    else:
        raise TypeError("Can't perform quantization. Unknown quantization scheme: {}".format(qscheme))
        return

    scale = (tmax - tmin)/scale_denom 
    return scale, zero_point 
    
    
    
def quantize_qint(tensor,\
                  dtype = torch.qint8,\
                  qscheme = torch.per_tensor_affine,\
                  dim = None,\
                  return_scale_zeropoint = False):
    """Converts a float `tensor` to quantized tensor with `scale` and `zero_point`
    computed via function `get_scale_zeropoint`.
    
    Parameters
    ----------
    tensor : Tensor
        Float tensor to quantize.
    dtype : ``torch.dtype``, default is ``torch.qint8``
        The desired data type of returned tensor.
        Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``.
    qscheme : quantization scheme, default is ``torch.per_tensor_affine``
        Has to be one of: ``torch.per_tensor_affine``, ``torch.per_tensor_symmetric``, ``torch.per_channel_affine``, ``torch.per_channel_symmetric``
    dim : int or None, default is None
        If dim is not None, along the dimension `dim` the values in the `tensor` are scaled and offset by a different value (effectively the scale and offset become vectors).
        If dim is None, all values in the `tensor` are scaled and offset by the same value.
    return_scale_zeropoint : bool, default False
        Activate return of scale and zero_point.
            
    Returns
    -------
    Quantized Tensor
        float version of quantized `tensor`.
        
    scale
        Scale to apply in quantization formula.
    zero_point
        Offset in integer value that maps to float zero.          
    """
    
    
    scale, zero_point = get_scale_zeropoint(tensor,\
                                            dtype = dtype,\
                                            qscheme = qscheme,\
                                            dim = dim)
    
    if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
        qtensor = torch.quantize_per_channel(tensor,\
                                             scales=scale, zero_points = zero_point,\
                                             dtype = dtype, axis = dim)
    
    elif qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
        qtensor = torch.quantize_per_tensor(tensor,\
                                            scale=scale, zero_point = zero_point,\
                                            dtype = dtype)
    else:
        raise TypeError("Can't perform quantization. Unknown quantization scheme: {}".format(qscheme))
        return
    
    
    if return_scale_zeropoint:
        return qtensor.dequantize(), scale, zero_point
    
    else:
        return qtensor.dequantize()

