from .. import backend as T
from ..cp_tensor import CPTensor
from ..cp_tensor import cp_normalize
import numpy as np

# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>


def vonNeumann_entropy(tensor, offset=1e-12):
    """Returns the von Neumann entropy of the tensor.

    Parameters
    ----------
    tensor : (matrix or CP tensor)
        Data structure
    offset : (float)
        Minimum value at which to not consider eigenvalue zero to avoid singularities.

    Returns
    -------
    float
    """
    if isinstance(tensor, CPTensor):
        eig_vals = cp_normalize(tensor).weights

    else:
        eig_vals = T.eigh(tensor)[0]

    eig_vals = eig_vals[eig_vals > offset]


    return float(-T.sum(T.log2(eig_vals) * eig_vals))

