
def align_tensors(permutation_matrix):
    """
    The order of the rows in the matrix is adjusted such that the highest values are on the main
    diagonal.

    :param permutation_matrix: pre-calculated permutation matrix, values can be in the range [-1, 1]
    :return order: order of the rows such that the highest values are on the main diagonal
                    -> matrix = matrix[order, :]
    """

    order = tl.argmax(permutation_matrix, axis=0)
    uniq, _, counts = np.unique(order, return_index=True, return_counts=True)
    uniq = tl.tensor(uniq)
    counts = tl.tensor(counts)
    if len(uniq) < len(order):
        non_uniq = uniq[counts > 1][0]
        # idea works as long as there are only two equal values in order (non_uniq_idx has length 2)
        non_uniq_idx = tl.tensor(np.where(order == non_uniq)[0])
        wrong_idx = non_uniq_idx[tl.argmin(permutation_matrix[np.full_like(non_uniq_idx, non_uniq),
                                                              non_uniq_idx])]
        order[wrong_idx] = np.setdiff1d(np.arange(len(order)), order)[0]

    return order


def factor_match_score_3d(kruskal_tensor_3d_true, kruskal_tensor_2d_true, kruskal_tensor_3d_pred,
                          kruskal_tensor_2d_pred):
    """
    Calculates factor match score (FMS) for comparing true and estimated matrices.

    Assume a 3rd order tensor and a matrix are coupled in the 1th mode with
    X = [[A, B, C]] and Y = [[A, V]]
    The FMS shows how close the estimation of A, B, C, V is to the true matrices.
    An FMS of 1 is a perfect match. The worst value is 0.

    Formula for FMS according to Equation 4
    @article{acar2011all,
      title={All-at-once optimization for coupled matrix and tensor factorizations},
      author={Acar, Evrim and Kolda, Tamara G and Dunlavy, Daniel M},
      journal={arXiv preprint arXiv:1105.3422},
      year={2011}
    }

    Notes
    -----
    In a first step, the tensors are aligned. Therefore, <a_i, a_pred_j> * <b_i, b_pred_j> *
    <c_i, c_pred_j> is calculated for all i and j, not only for i=j. If the i-th vector of the
    true factor matrix belongs to the j-th vector of the predicted matrix, the result is close to
    1, otherwise very small. The order of the predicted factors is adjusted in the align_tensors
    function, so that the i-th true factor belongs to the i-th predicted factor, i.e. that
    (A.T @ A_pred) * (B.T @ B_pred) * (C.T @ C_pred) consists of values close to 1 on its main
    diagonal.

    Parameters
    ----------
    kruskal_tensor_3d_true : Kruskal tensor
        true tensor X = [[A, B, C]]
    kruskal_tensor_2d_true : Kruskal tensor
        true matrix Y = [[A, V]]
    kruskal_tensor_3d_pred : Kruskal tensor
        tensor consisting of predicted factor matrices A_pred, B_pred, C_pred: X_pred = [[A_pred,
        B_pred, C_pred]]
    kruskal_tensor_2d_pred : Kruskal tensor
        matrix consisting of predicted factor matrices A_pred, V_pred: Y_pred = [[A_pred, V_pred]]

    Returns
    -------
    factor match score : float
        value between 0 and 1 that shows how good the estimation is, 1 is perfect

    """

    lambda_, [A, B, C] = kruskal_normalise(kruskal_tensor_3d_true)
    alpha_, [a, V] = kruskal_normalise(kruskal_tensor_2d_true)
    xi = lambda_ + alpha_

    lambda_pred, [A_pred, B_pred, C_pred] = kruskal_normalise(kruskal_tensor_3d_pred)
    alpha_pred, [a_pred, V_pred] = kruskal_normalise(kruskal_tensor_2d_pred)
    xi_pred = lambda_pred + alpha_pred

    # take care of permutation of normalized factor matrices and of sign of weights (not mentioned
    # in paper, but necessary)
    product = tl.dot(tl.transpose(A), A_pred) * tl.dot(tl.transpose(B), B_pred) * tl.dot(
        tl.transpose(C), C_pred)
    perm = product * tl.sign(tl.dot(tl.reshape(lambda_, (-1, 1)), tl.reshape(lambda_pred, (1, -1))))
    order = align_tensors(perm)

    product = product[order, :]  # product[r,r] now corresponds to
    # (a_r.T @ a_pred_r) / (||(a_r|| ||a_pred_r||) * (b_r.T @ b_pred_r) / (||(b_r|| ||b_pred_r||) *
    # (c_r.T @ c_pred_r) / (||(c_r|| ||c_pred_r||)
    V = V[:, order]
    xi = xi[order]

    # calculate factor match score
    rank = A.shape[1]
    FMS = tl.zeros(rank)
    for r in range(rank):
        FMS[r] = (1 - tl.abs(xi[r] - xi_pred[r]) / max(xi[r], xi_pred[r])) * tl.abs(
            product[r, r] * tl.dot(V[:, r], V_pred[:, r]))

    return tl.min(FMS)
