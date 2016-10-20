import numpy as np

# Author: Jean Kossaifi


def norm(tensor, order):
    """Computes the l-order norm of tensor

    Parameters
    ----------
    tensor : ndarray
    order : int

    Returns
    -------
    float
        l-order norm of tensor
    """
    if order == 1:
        return np.sum(np.abs(tensor))
    elif order == 2:
        return np.sqrt(np.sum(tensor**2))
    else:
        return np.sum(np.abs(tensor)**order)**(1/order)
