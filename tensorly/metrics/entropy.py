import tensorly as tl
from .. import backend as T
from ..cp_tensor import CPTensor, cp_normalize
from ..tt_tensor import tt_to_tensor

# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>


def vonneumann_entropy(tensor):
    """Returns the von Neumann entropy of the tensor.

    Parameters
    ----------
    tensor : (matrix)
        Data structure

    Returns
    -------
    von_neumann_entropy : order-0 tensor
    """
    eig_vals = T.eigh(tensor)[0]
    eps = tl.eps(eig_vals.dtype)
    eig_vals = eig_vals[eig_vals >  eps]

    return -T.sum(T.log2(eig_vals) * eig_vals)

def tt_mps_entanglement_entropy(tensor, boundary):
    """Returns the entanglement entropy of an MPS paritioned at boundary in TT tensor form. Assumes
    a traditional and single MPS, that is, a linear pure state.

    Parameters
    ----------
    tensor : (TT tensor)
        Data structure
    boundary : (int)
        Qubit at which to partition system.

    Returns
    -------
    tt_mps_entanglement_entropy : order-0 tensor
    """
    partial_mps = tensor[boundary]
    dims = partial_mps.shape
    partial_mps = tl.reshape(partial_mps, (1, dims[0]*dims[1], dims[2]))
    partial_mps = tt_to_tensor([partial_mps] + tensor[boundary+1::])
    partial_mps = tl.reshape(partial_mps, (dims[0]*dims[1], -1))
    _, eig_vals, _ = tl.partial_svd(partial_mps, min(partial_mps.shape))
    eig_vals = eig_vals**2
    eps = tl.eps(eig_vals.dtype)
    eig_vals = eig_vals[eig_vals > eps]

    return -T.sum(T.log2(eig_vals) * eig_vals)


def tt_vonneumann_entropy(tensor):
    """Returns the von Neumann entropy of the TT tensor.

    Parameters
    ----------
    tensor : (TT tensor)
        Data structure

    Returns
    -------
    tt_von_neumann_entropy : order-0 tensor
    """
    square_dim = int(tl.sqrt(tl.prod(tl.tensor(tensor.shape))))
    tensor = tl.reshape(tt_to_tensor(tensor), (square_dim, square_dim))

    return vonneumann_entropy(tensor)


def cp_vonneumann_entropy(tensor):
    """Returns the von Neumann entropy of the CP tensor. 

    Parameters
    ----------
    tensor : (CP tensor)
        Data structure

    Returns
    -------
    cp_von_neumann_entropy : order-0 tensor
    """
    eig_vals = cp_normalize(tensor).weights
    eps = tl.eps(eig_vals.dtype)
    eig_vals = eig_vals[eig_vals > eps]

    return -T.sum(T.log2(eig_vals) * eig_vals)
