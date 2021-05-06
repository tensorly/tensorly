from scipy.optimize import linear_sum_assignment
from .. import backend as T


def congruence_coefficient(matrix1, matrix2, absolute_value=True):
    """Compute the optimal mean (Tucker) congruence coefficient between the columns of two matrices.

    Another name for the congruence coefficient is the cosine similarity.

    The congruence coefficient between two vectors, :math:`\\mathbf{v}_1, \\mathbf{v}_2`, is given by

    .. math::

        \\frac{\\mathbf{v}_1^T \\mathbf{v}_1^T}{\\|\\mathbf{v}_1^T\\| \\|\\mathbf{v}_1^T\\|}

    When we compute the congruence between two matrices, we find the optimal permutation of
    the columns and return the mean congruence and the permutation.

    Parameters
    ----------
    matrix1 : tensorly.Tensor
    matrix2 : tensorly.Tensor
    absolute_value : bool
        Whether to take the absolute value of all vector products before finding the optimal permutation.

    Returns
    -------
    congruence : float
    permutation : list
    """
    num_cols = T.shape(matrix1)[1]
    if T.shape(matrix1) != T.shape(matrix2):
        raise ValueError("Matrices must have same shape")

    matrix1 = matrix1/T.norm(matrix1, axis=0)
    matrix2 = matrix2/T.norm(matrix2, axis=0)
    all_congruences = T.dot(T.transpose(matrix1), matrix2)
    if absolute_value:
        all_congruences = T.abs(all_congruences)
    all_congruences = T.to_numpy(all_congruences)
    row_ind, col_ind = linear_sum_assignment(-all_congruences)   # Use -corr because scipy didn't doesn't support maximising prior to v1.4
    indices = dict(zip(row_ind, col_ind))
    permutation = [indices[i] for i in range(num_cols)]
    return all_congruences[row_ind, col_ind].mean(), permutation