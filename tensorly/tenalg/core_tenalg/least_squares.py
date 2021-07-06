from numpy.linalg import lstsq


def solve_least_squares(A, B):
    """

    Parameters
    ----------
    A : np.ndarray
        dimensions: M x N
    B : np.ndarray
        dimensions: M x K

    Returns
    -------

    """

    #TODO: implement leastsquares for all backends
    return lstsq(A, B)[0]
