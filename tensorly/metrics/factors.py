from scipy.optimize import linear_sum_assignment
from .. import backend as T


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
    # Check if matrix1 and matrix2 are lists of the same length
    if isinstance(matrix1, list):
        if not isinstance(matrix2, list) or len(matrix1) != len(matrix2):
            raise ValueError("Input lists of matrices must have the same length")
    else:
        matrix1 = [matrix1]
        matrix2 = [matrix2]
    all_congruences_list = []
    # Store an arbitrary column dimension, they should all be the same
    num_col_mat1 = T.shape(matrix1[0])[1]
    for mat1, mat2 in zip(matrix1, matrix2):
        num_cols = T.shape(mat1)[1]
        if T.shape(mat1) != T.shape(mat2):
            raise ValueError("Matrices must have same shape")
        if T.shape(mat1)[1] != num_col_mat1 or T.shape(mat2)[1] != num_col_mat1:
            # check that all matrices have the same number of columns
            raise ValueError("Matrices must have the same number of columns")

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
    permutation = [indices[i] for i in range(num_cols)]
    return all_congruences[row_ind, col_ind].mean(), permutation
