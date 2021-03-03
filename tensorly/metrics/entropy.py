from .. import backend as T
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

    if tensor.__class__.__name__ == 'CPTensor':
        weights = tensor[0]
        if len(weights) == 1:
            return 0
        if np.allclose(weights, np.ones(len(weights))):
            weights = cp_normalize(tensor)[0]
        return T.sum(-weights*T.log2(weights))

    eig_vals = T.eigh(tensor)[0]
    eig_vals = eig_vals[eig_vals > offset]
    return -T.sum(T.log2(eig_vals)*eig_vals)

