from ... import backend as T
from ..tenalg_utils import _validate_khatri_rao

# Author: Jean Kossaifi

# License: BSD 3 clause


def khatri_rao(matrices, weights=None, skip_matrix=None, reverse=False, mask=None):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.
        (see [1]_ for more details).

        If one matrix only is given, that matrix is directly returned.

    Parameters
    ----------
    matrices : 2D-array list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)
    
    weights : 1D-array
        array of weights for each rank, of length m, the number of column of the factors
        (i.e. m == factor[i].shape[1] for any factor)

    skip_matrix : None or int, optional, default is None
        if not None, index of a matrix to skip

    reverse : bool, optional
        if True, the order of the matrices is reversed

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the product.

    Notes
    -----
    Mathematically:

    .. math::
         \\text{If every matrix } U_k \\text{ is of size } (I_k \\times R),\\\\
         \\text{Then } \\left(U_1 \\bigodot \\cdots \\bigodot U_n \\right) \\text{ is of size } (\\prod_{k=1}^n I_k \\times R)

    A more intuitive but slower implementation is::

        kr_product = np.zeros((n_rows, n_columns))
        for i in range(n_columns):
            cum_prod = matrices[0][:, i]  # Accumulates the khatri-rao product of the i-th columns
            for matrix in matrices[1:]:
                cum_prod = np.einsum('i,j->ij', cum_prod, matrix[:, i]).ravel()
            # the i-th column corresponds to the kronecker product of all the i-th columns of all matrices:
            kr_product[:, i] = cum_prod

        return kr_product


    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    matrices, n_col = _validate_khatri_rao(
        matrices, skip_matrix=skip_matrix, reverse=reverse
    )

    # Khatri-rao of only one matrix: just return that matrix
    if len(matrices) == 1:
        return matrices[0]

    for i, e in enumerate(matrices[1:]):
        if not i:
            if weights is None:
                res = matrices[0]
            else:
                res = matrices[0] * T.reshape(weights, (1, -1))
        s1, s2 = T.shape(res)
        s3, s4 = T.shape(e)

        a = T.reshape(res, (s1, 1, s2))
        b = T.reshape(e, (1, s3, s4))
        res = T.reshape(a * b, (-1, n_col))

    m = T.reshape(mask, (-1, 1)) if mask is not None else 1

    return res * m
