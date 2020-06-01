from ... import backend as T

# Author: Jean Kossaifi

# License: BSD 3 clause



def kronecker(matrices, skip_matrix=None, reverse=False):
    """Kronecker product of a list of matrices

        For more details, see [1]_

    Parameters
    ----------
    matrices : ndarray list

    skip_matrix : None or int, optional, default is None
        if not None, index of a matrix to skip

    reverse : bool, optional
        if True, the order of the matrices is reversed

    Returns
    -------
    kronecker_product: matrix of shape ``(prod(n_rows), prod(n_columns)``
        where ``prod(n_rows) = prod([m.shape[0] for m in matrices])``
        and ``prod(n_columns) = prod([m.shape[1] for m in matrices])``

    Notes
    -----
    Mathematically:

    .. math::
         \\text{If every matrix } U_k \\text{ is of size } (I_k \\times J_k),\\\\
         \\text{Then } \\left(U_1 \\otimes \\cdots \\otimes U_n \\right) \\text{ is of size } (\\prod_{k=1}^n I_k \\times \\prod_{k=1}^n J_k)

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    if reverse:
        order = -1
    else:
        order = 1

    for i, matrix in enumerate(matrices[::order]):
        if not i:
            res = matrix
        else:
            res = T.kron(res, matrix)
    return res

