import warnings
from typing import Literal
import tensorly as tl
from .proximal import soft_thresholding

# Authors: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>
#          Meraj Hashemizadeh <merajhse@mila.quebec>
#          Aaron Meyer <tensorly@ameyer.me>

# License: BSD 3 clause


def svd_flip(U, V, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    This function is borrowed from scikit-learn/utils/extmath.py
    Parameters
    ----------
    U : ndarray
        u and v are the output of SVD
    V : ndarray
        u and v are the output of SVD
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of U, rows of V
        max_abs_cols = tl.argmax(tl.abs(U), axis=0)
        signs = tl.sign(
            tl.tensor(
                [U[i, j] for (i, j) in zip(max_abs_cols, range(tl.shape(U)[1]))],
                **tl.context(U),
            )
        )
        U = U * signs
        if tl.shape(V)[0] > tl.shape(U)[1]:
            signs = tl.concatenate(
                (signs, tl.ones(tl.shape(V)[0] - tl.shape(U)[1], **tl.context(V)))
            )
        V = V * signs[: tl.shape(V)[0]][:, None]
    else:
        # rows of V, columns of U
        max_abs_rows = tl.argmax(tl.abs(V), axis=1)
        signs = tl.sign(
            tl.tensor(
                [V[i, j] for (i, j) in zip(range(tl.shape(V)[0]), max_abs_rows)],
                **tl.context(V),
            )
        )
        V = V * signs[:, None]
        if tl.shape(U)[1] > tl.shape(V)[0]:
            signs = tl.concatenate(
                (signs, tl.ones(tl.shape(U)[1] - tl.shape(V)[0], **tl.context(V)))
            )
        U = U * signs[: tl.shape(U)[1]]

    return U, V


def make_svd_non_negative(tensor, U, S, V, nntype=True):
    """Use NNDSVD method to transform SVD results into a non-negative form. This
    method leads to more efficient solving with NNMF [1].

    Parameters
    ----------
    tensor : tensor being decomposed
    U, S, V: SVD factorization results
    nntype : {'nndsvd', 'nndsvda'}
        Whether to fill small values with 0.0 (nndsvd), or the tensor mean (nndsvda, default).

    [1]: Boutsidis & Gallopoulos. Pattern Recognition, 41(4): 1350-1362, 2008.
    """
    if nntype is True:
        nntype = "nndsvda"

    # NNDSVD initialization
    W = tl.zeros_like(U)
    H = tl.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W = tl.index_update(W, tl.index[:, 0], tl.sqrt(S[0]) * tl.abs(U[:, 0]))
    H = tl.index_update(H, tl.index[0, :], tl.sqrt(S[0]) * tl.abs(V[0, :]))

    for j in range(1, min(tl.shape(U)[1], tl.shape(V)[0])):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = tl.clip(x, a_min=0.0), tl.clip(y, a_min=0.0)
        x_n, y_n = tl.abs(tl.clip(x, a_max=0.0)), tl.abs(tl.clip(y, a_max=0.0))

        # and their norms
        x_p_nrm, y_p_nrm = tl.norm(x_p), tl.norm(y_p)
        x_n_nrm, y_n_nrm = tl.norm(x_n), tl.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = tl.sqrt(S[j] * sigma)
        W = tl.index_update(W, tl.index[:, j], lbd * u)
        H = tl.index_update(H, tl.index[j, :], lbd * v)

    # After this point we no longer need H
    eps = tl.eps(tensor.dtype)

    if nntype == "nndsvd":
        W = soft_thresholding(W, eps)
        H = soft_thresholding(H, eps)
    elif nntype == "nndsvda":
        avg = tl.mean(tensor)
        W = tl.where(W < eps, tl.ones(tl.shape(W), **tl.context(W)) * avg, W)
        H = tl.where(H < eps, tl.ones(tl.shape(H), **tl.context(H)) * avg, H)
    else:
        raise ValueError(
            f'Invalid nntype parameter: got {nntype} instead of one of ("nndsvd", "nndsvda")'
        )

    return W, H


def randomized_range_finder(A, n_dims, n_iter=2, random_state=None):
    """Computes an orthonormal matrix (Q) whose range approximates the range of A,  i.e., Q Q^H A ≈ A

    Parameters
    ----------
    A : 2D-array
    n_dims : int, dimension of the returned subspace
    n_iter : int, number of power iterations to conduct (default = 2)
    random_state: {None, int, np.random.RandomState}

    Returns
    -------
    Q : 2D-array
        of shape (A.shape[0], min(n_dims, A.shape[0], A.shape[1]))

    Notes
    -----
    This function is implemented based on Algorith 4.4 in `Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions`
    - Halko et al (2009)
    """
    rng = tl.check_random_state(random_state)
    dim_1, dim_2 = tl.shape(A)
    Q = tl.tensor(rng.normal(size=(dim_2, n_dims)), **tl.context(A))
    Q, _ = tl.qr(tl.dot(A, Q))

    # Perform power iterations when spectrum decays slowly
    A_H = tl.conj(tl.transpose(A))
    for i in range(n_iter):
        Q, _ = tl.qr(tl.dot(A_H, Q))
        Q, _ = tl.qr(tl.dot(A, Q))

    return Q


def svd_checks(matrix, n_eigenvecs=None):
    """Runs common checks to all of the SVD methods.

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    n_eigenvecs : int
        the number of eigenvectors to solve for
    min_dim : int
        the minimum dimension of matrix
    max_dim : int
        the maximum dimension of matrix
    """
    # Check that matrix is... a matrix!
    if tl.ndim(matrix) != 2:
        raise ValueError(f"matrix be a matrix. matrix.ndim is {tl.ndim(matrix)} != 2")

    dim_1, dim_2 = tl.shape(matrix)
    min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

    if n_eigenvecs is None:
        n_eigenvecs = max_dim

    if n_eigenvecs > max_dim:
        warnings.warn(
            f"Trying to compute SVD with n_eigenvecs={n_eigenvecs}, which is larger "
            f"than max(matrix.shape)={max_dim}. Setting n_eigenvecs to {max_dim}."
        )
        n_eigenvecs = max_dim

    return n_eigenvecs, min_dim, max_dim


def truncated_svd(matrix, n_eigenvecs=None, **kwargs):
    """Computes a truncated SVD on `matrix` using the backends's standard SVD

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    U : 2D-array
        of shape (matrix.shape[0], n_eigenvecs)
        contains the right singular vectors
    S : 1D-array
        of shape (n_eigenvecs, )
        contains the singular values of `matrix`
    V : 2D-array
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    n_eigenvecs, min_dim, _ = svd_checks(matrix, n_eigenvecs=n_eigenvecs)
    full_matrices = True if n_eigenvecs > min_dim else False
    U, S, V = tl.svd(matrix, full_matrices=full_matrices)
    return U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]


def symeig_svd(matrix, n_eigenvecs=None, **kwargs):
    """Computes a truncated SVD on `matrix` using symeig

        Uses symeig on matrix.T.dot(matrix) or its transpose

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return
    **kwargs : optional
        kwargs are used to absorb the difference of parameters among the other SVD functions

    Returns
    -------
    U : 2D-array
        of shape (matrix.shape[0], n_eigenvecs)
        contains the right singular vectors
    S : 1D-array
        of shape (n_eigenvecs, )
        contains the singular values of `matrix`
    V : 2D-array
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    n_eigenvecs, _, _ = svd_checks(matrix, n_eigenvecs=n_eigenvecs)

    dim_1, dim_2 = tl.shape(matrix)

    if dim_1 > dim_2:
        S, U = tl.eigh(tl.dot(matrix, tl.transpose(matrix)))
        S = tl.sqrt(tl.clip(S, tl.eps(S.dtype)))
        V = tl.dot(tl.transpose(matrix), U / tl.reshape(S, (1, -1)))
    else:
        S, V = tl.eigh(tl.dot(tl.transpose(matrix), matrix))
        S = tl.sqrt(tl.clip(S, tl.eps(S.dtype)))
        U = tl.dot(matrix, V) / tl.reshape(S, (1, -1))

    U, S, V = (
        tl.flip(U, axis=1),
        tl.flip(S),
        tl.flip(tl.transpose(V), axis=0),
    )
    return (
        U[:, : min(dim_1, n_eigenvecs)],
        S[: min(dim_1, dim_2, n_eigenvecs)],
        V[: min(dim_2, n_eigenvecs), :],
    )


def randomized_svd(
    matrix,
    n_eigenvecs=None,
    n_oversamples=5,
    n_iter=2,
    random_state=None,
    **kwargs,
):
    """Computes a truncated randomized SVD.

    If `n_eigenvecs` is specified, sparse eigendecomposition is used on
    either matrix.dot(matrix.T) or matrix.T.dot(matrix).

    Parameters
    ----------
    matrix : tensor
        A 2D tensor.
    n_eigenvecs : int, optional, default is None
        If specified, number of eigen[vectors-values] to return.
    n_oversamples: int, optional, default = 5
        rank overestimation value for finiding subspace with better allignment
    n_iter: int, optional, default = 2
        number of power iterations for the `randomized_range_finder` subroutine
    random_state: {None, int, np.random.RandomState}
    **kwargs : optional
        kwargs are used to absorb the difference of parameters among the other SVD functions

    Returns
    -------
    U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
        Contains the right singular vectors
    S : 1-D tensor, shape (n_eigenvecs, )
        Contains the singular values of `matrix`
    V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
        Contains the left singular vectors

    Notes
    -----
    This function is implemented based on Algorith 5.1 in `Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions`
    - Halko et al (2009)
    """
    n_eigenvecs, min_dim, max_dim = svd_checks(matrix, n_eigenvecs=n_eigenvecs)

    dim_1, dim_2 = tl.shape(matrix)
    n_dims = min(n_eigenvecs + n_oversamples, max_dim)

    if (
        dim_1 > dim_2
        and n_eigenvecs > min(min_dim, n_dims)
        or dim_1 < dim_2
        and n_eigenvecs < min(min_dim, n_dims)
    ):
        # transpose matrix to keep the reduced matrix shape minimal
        matrix_T = tl.transpose(matrix)
        Q = randomized_range_finder(
            matrix_T, n_dims=n_dims, n_iter=n_iter, random_state=random_state
        )
        Q_H = tl.conj(tl.transpose(Q))
        matrix_reduced = tl.transpose(tl.dot(Q_H, matrix_T))
        U, S, V = truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
        V = tl.dot(V, tl.transpose(Q))
    else:
        Q = randomized_range_finder(
            matrix, n_dims=n_dims, n_iter=n_iter, random_state=random_state
        )
        Q_H = tl.conj(tl.transpose(Q))
        matrix_reduced = tl.dot(Q_H, matrix)
        U, S, V = truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
        U = tl.dot(Q, U)

    return U, S, V


SVD_FUNS = ["truncated_svd", "symeig_svd", "randomized_svd"]
SVD_TYPES = Literal["truncated_svd", "symeig_svd", "randomized_svd"]


def svd_interface(
    matrix,
    method="truncated_svd",
    n_eigenvecs=None,
    flip_sign=True,
    u_based_flip_sign=True,
    non_negative=None,
    mask=None,
    n_iter_mask_imputation=5,
    **kwargs,
):
    """Dispatching function to various SVD algorithms, alongside additional
    properties such as resolving sign invariance, imputation, and non-negativity.

    Parameters
    ----------
    matrix : tensor
        A 2D tensor.
    method : str, default is 'truncated_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS or a callable.
    n_eigenvecs : int, optional, default is None
        If specified, number of eigen[vectors-values] to return.
    flip_sign : bool, optional, default is True
        Whether to resolve the sign indeterminacy of SVD.
    u_based_flip_sign : bool, optional, default is True
        Whether the sign indeterminacy should be resolved using U (vs. V).
    non_negative : bool, optional, default is False
        Whether to make the SVD results non-negative.
    nn_type : str, default is 'nndsvd'
        Algorithm to use for converting U to be non-negative.
    mask : tensor, default is None.
        Array of booleans with the same shape as ``matrix``. Should be 0 where
        the values are missing and 1 everywhere else. None if nothing is missing.
        Imputation is done by iterative low rank approximation, so n_eigenvecs should be provided
        and be lower than the rank of the matrix.
    n_iter_mask_imputation : int, default is 5
        Number of repetitions to apply in missing value imputation.
    **kwargs : optional
        Arguments passed along to individual SVD algorithms.

    Returns
    -------
    U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
        Contains the right singular vectors of `matrix`
    S : 1-D tensor, shape (n_eigenvecs, )
        Contains the singular values of `matrix`
    V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
        Contains the left singular vectors of `matrix`
    """

    if method == "truncated_svd":
        svd_fun = truncated_svd
    elif method == "symeig_svd":
        svd_fun = symeig_svd
    elif method == "randomized_svd":
        svd_fun = randomized_svd
    elif callable(method):
        svd_fun = method
    else:
        raise ValueError(
            f"Got svd={method}. However, the possible choices are {SVD_FUNS} or to pass a callable."
        )

    U, S, V = svd_fun(matrix, n_eigenvecs=n_eigenvecs, **kwargs)

    if mask is not None and n_eigenvecs is not None:
        for _ in range(n_iter_mask_imputation):
            # Workaround to avoid needing fill_diagonal
            St = tl.eye(tl.shape(U)[1], tl.shape(V)[0], **tl.context(matrix))
            for i in range(tl.shape(S)[0]):
                St = tl.index_update(St, tl.index[i, i], S[i])

            matrix = matrix * mask + (U @ St @ V) * (1 - mask)
            U, S, V = svd_fun(matrix, n_eigenvecs=n_eigenvecs, **kwargs)

    if flip_sign:
        U, V = svd_flip(U, V, u_based_decision=u_based_flip_sign)

    if non_negative is not False and non_negative is not None:
        U, V = make_svd_non_negative(matrix, U, S, V, non_negative)

    return U, S, V
