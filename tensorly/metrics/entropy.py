import math
import tensorly as tl
from .. import backend as T
from ..cp_tensor import cp_normalize
from ..tt_tensor import tt_to_tensor
from ..utils import prod

# Authors: Taylor Lee Patti <taylorpatti@g.harvard.edu>
#          Jean Kossaifi

def vonneumann_entropy(tensor):
    """Returns the von Neumann entropy of a density matrix (2-mode, square) tensor (matrix).

    Parameters
    ----------
    tensor : Non-decomposed tensor with indices whose shapes are all a factor of two (represent one or more qubits)

    Returns
    -------
    von_neumann_entropy : order-0 tensor

    Notes
    -----
    The von Neumann entropy is :math:`- \\sum_i p_i ln(p_i)`, 
    where p_i are the probabilities that each state is occupied 
    (the eigenvalues of the density matrix).
    """
    square_dim = int(math.sqrt(prod(tensor.shape)))
    tensor = tl.reshape(tensor, (square_dim, square_dim))
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

    return vonneumann_entropy(tt_to_tensor(tensor))


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
