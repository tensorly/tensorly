from scipy.optimize import linear_sum_assignment
from .. import backend as T
import numpy as np


def congruence_coefficient(matrix1, matrix2, absolute_value=True):
    """Compute the optimal mean (Tucker) congruence coefficient between the columns of two matrices.

    Another name for the congruence coefficient is the cosine similarity.

    The congruence coefficient between two vectors, :math:`\\mathbf{v}_1, \\mathbf{v}_2`, is given by

    .. math::

        \\frac{\\mathbf{v}_1^T \\mathbf{v}_1^T}{\\|\\mathbf{v}_1^T\\| \\|\\mathbf{v}_1^T\\|}

    When we compute the congruence between two matrices, we find the optimal permutation of
    the columns and return the mean congruence and the permutation. The output permutation is the one
    that permutes the columns of matrix2 onto the closest columns in matrix1.

    If a list of matrices is provided for each input, we define the congruence coefficient as the
    product of the absolute values of pairs of matrices. The lists must therefore have the same size.
    The output permutation also applies to each matrix of the lists.

    Parameters
    ----------
    matrix1 : tensorly.Tensor or list of tensorly.Tensor
    matrix2 : tensorly.Tensor of list of tensorly.Tensor to permute.
    absolute_value : bool
        Whether to take the absolute value of all vector products before finding the optimal permutation.

    Returns
    -------
    congruence : float
    permutation : list
    """
    if T.is_tensor(matrix1):
        matrix1 = [matrix1]
    if T.is_tensor(matrix2):
        matrix2 = [matrix2]
    # Check if matrix1 and matrix2 are lists of the same length
    if len(matrix1) != len(matrix2):
        raise ValueError("Input lists of matrices must have the same length")
    all_congruences_list = []
    # Check if all matrices have the same number of columns
    columns = [T.shape(m)[1] for m in matrix1] + [T.shape(m)[1] for m in matrix2]
    if len(np.unique(columns)) > 1:
        raise ValueError("All matrices must have the same number of columns")
    for mat1, mat2 in zip(matrix1, matrix2):
        if T.shape(mat1)[0] != T.shape(mat2)[0]:
            raise ValueError("Pairs of matrices must have the same number of rows")
        # Check if any norm is exactly zero to avoid singularity
        if T.prod(T.norm(mat1, axis=0)) == 0 or T.prod(T.norm(mat2, axis=0)) == 0:
            raise ValueError("Columns of all matrices should have nonzero l2 norm")
        mat1 = mat1 / T.norm(mat1, axis=0)
        mat2 = mat2 / T.norm(mat2, axis=0)
        all_congruences_list.append(T.dot(T.transpose(mat1), mat2))
        if absolute_value:
            all_congruences_list[-1] = T.abs(all_congruences_list[-1])
        all_congruences_list[-1] = T.to_numpy(all_congruences_list[-1])
    all_congruences = 1
    for congruence in all_congruences_list:
        all_congruences *= congruence
    row_ind, col_ind = linear_sum_assignment(
        -all_congruences
    )  # Use -corr because scipy didn't doesn't support maximising prior to v1.4
    indices = dict(zip(row_ind, col_ind))
    permutation = [indices[i] for i in range(T.shape(matrix1[0])[1])]
    return all_congruences[row_ind, col_ind].mean(), permutation
