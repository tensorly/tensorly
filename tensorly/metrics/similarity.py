import tensorly as tl

# Authors: Hratch Baghdassarian <hmbaghdassarian@gmail.com>, Erick Armingol <earmingol14@gmail.com>
# similarity metrics for tensor decompsoitions


def correlation_index(factors_1: list, factors_2: list, tol: float = 5e-16) -> float:
    """CorrIndex implementation to assess tensor decomposition outputs. 
    From Sobhani et al 2022 (https://doi.org/10.1016/j.sigpro.2022.108457). 
    Metric is scaling and column-permutation invariant.

    Parameters
    ----------
    factors_1 : list
        The loading/factor matrices [A^1 ... A^R] for a low-rank tensor from its factors, output from first decomposition
    factors_2 : list
        The loading/factor matrices [A^1 ... A^R] for a low-rank tensor from its factors, output from second decomposition
    tol : float, optional
        precision threshold below which to call the CorrIndex score 0, by default 5e-16

    Returns
    -------
    score : float
         CorrIndex metric [0,1]; lower score indicates higher similarity between matrices
    """
    # check input factors shape
    for factors in [factors_1, factors_2]:
        if len({tl.shape(A)[1]for A in factors}) != 1:
            raise ValueError('Factors should be a list of loading matrices of the same rank')

    # vertically stack loading matrices -- shape sum(tensor.shape)xR)
    X_1 = tl.concatenate(factors_1)
    X_2 = tl.concatenate(factors_2)

    if tl.shape(X_1) != tl.shape(X_2):
        raise ValueError('Factor matrices should be of the same shapes')

    # normalize columns to L2 norm - even if ran decomposition with normalize_factors=True
    col_norm_1 = tl.norm(X_1, axis=0)
    col_norm_2 = tl.norm(X_2, axis=0)
    if tl.any(col_norm_1 == 0) or tl.any(col_norm_2 == 0):
        raise ValueError('Column norms must be non-zero')
    X_1 = X_1/col_norm_1
    X_2 = X_2/col_norm_2

    # generate the correlation index input
    c_prod_mtx = tl.abs(tl.matmul(tl.conj(X_1.T), X_2))

    # correlation index scoring
    n_elements = tl.shape(c_prod_mtx)[1] * tl.shape(c_prod_mtx)[0] #TODO: N notation in equation 8 unclear, make sure this is correct
    score = (1/(n_elements)) * (tl.sum(tl.abs(tl.max(c_prod_mtx,1) - 1)) + tl.sum(tl.abs(tl.max(c_prod_mtx, 0) - 1)))
    if score < tol:
        score = 0

    return score
