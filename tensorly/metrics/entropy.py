import tensorly as tl
from .. import backend as T
from ..cp_tensor import CPTensor, cp_normalize
from ..tt_tensor import tt_to_tensor

# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>


def vonneumann_entropy(tensor):
    """Returns the von Neumann entropy of a density matrix (2-mode, square) tensor (matrix). 

    Parameters
    ----------
    tensor : (matrix)
        Data structure

    Returns
    -------
    von_neumann_entropy : order-0 tensor
    """
    try:
        eig_vals = T.eigh(tensor)[0]
    except:
    #All density matrices are Hermitian, here real. Hermitianize matrix if rounding/transformation
    #errors have occured.
        tensor = (tensor + tl.transpose(tensor))/2
        eig_vals = T.eigh(tensor)[0]
    eps = tl.eps(eig_vals.dtype)
    eig_vals = eig_vals[eig_vals >  eps]

    return -T.sum(T.log2(eig_vals) * eig_vals)


def tt_vonneumann_entropy(tensor):
    """Returns the von Neumann entropy of a density matrix (square matrix) in TT tensor form.

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
    """Returns the von Neumann entropy of a density matrix (square matrix) in CP tensor. 

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
