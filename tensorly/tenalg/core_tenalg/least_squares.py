import tensorly as tl


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

    return tl.lstsq(A, B)[0]
