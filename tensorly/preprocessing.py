from tensorly import backend as T

from .parafac2_tensor import Parafac2Tensor
from .tenalg.svd import svd_interface


def svd_compress_tensor_slices(
    tensor_slices, compression_threshold=0.0, max_rank=None, svd="truncated_svd"
):
    r"""Compress data with the SVD for running PARAFAC2.

    PARAFAC2 can be sped up massively for data where the number of rows in the tensor slices
    is much greater than their rank. In that case, we can compress the data by computing the
    SVD and fitting the PARAFAC2 model to the right singular vectors multiplied by the singular
    values. Then, we can "decompress" the decomposition by left-multiplying the :math:`B_i`-matrices
    by the left singular values to get a decomposition as if it was fitted to the uncompressed
    data. We can essentially think of this as running a PCA without centering the data for each
    tensor slice and fitting the PARAFAC2 model to the scores. Then, to get back the components,
    we left-multiply the :math:`B_i`-matrices with the loading matrices.

    [1]_ states that we can constrain our :math:`B_i`-matrices to lie in a given vector space,
    :math:`\mathscr{V}_i` by multiplying the data matrices with an orthogonal basis matrix that
    spans :math:`\mathscr{V}_i`. However, since we know that :math:`B_i` lie in the column space
    of :math:`X_i`, we can multiply the :math:`X_i`-matrices by an orthogonal matrix that spans
    :math:`\text{col}(X_i)` without affecting the fit of the model. Thus we can compress our data
    prior to fitting the PARAFAC2 model whenever the number of rows in our data matrices exceeds
    the number of columns (as the rank of :math:`\text{col}(X_i)` cannot exceed the number of rows).

    To implement this, we use the SVD to get an orthogonal basis for the column space of :math:`X_i`.
    Moreover, since :math:`S_i V_i^T = U_i^T X_i`, we can skip an additional matrix multiplication
    by fitting the model to :math:`S_i V_i^T`.

    Finally, we note that this approach can also be implemented by truncating the SVD. If an appropriate
    threshold is set, this will not affect the fitted model in any major form.

    .. note::
        This can be thought of as a simplified version of the DPAR approach for compressing PARAFAC2 models [2]_,
        which compresses all modes of :math:`\mathcal{X}` to fit an approximate PARAFAC2 model.

    Parameters
    ----------
    tensor_slices : list of matrices
        The data matrices to compress.
    compression_threshold : float (0 <= compression_threshold <= 1)
        Threshold at which the singular values should be truncated. Any singular value less than
        compression_threshold * s[0] is set to zero. Note that if this is nonzero, then the found
        components will likely be affected.
    max_rank : int
        The maximum rank to allow in the datasets after compression. This also serves to speed up
        the SVD calculation with matrices containing many rows and columns when paired with randomized
        SVD solving.
    svd : str, default is 'truncated_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS

    Returns
    -------
    list of matrices
        The score matrices, used to fit the PARAFAC2 model to.
    list of matrices
        The loading matrices, used to decompress the PARAFAC2 components after fitting
        to the scores.

    References
    ----------
    .. [1] Helwig, N. E. (2017). Estimating latent trends in multivariate longitudinal
           data via Parafac2 with functional and structural constraints. Biometrical
           Journal, 59(4), 783-803. doi: 10.1002/bimj.201600045

    .. [2] Jang JG, Kang U. Dpar2: Fast and scalable parafac2 decomposition for
           irregular dense tensors. 38th International Conference on Data Engineering
           (ICDE) 2022 May 9 (pp. 2454-2467). IEEE.

    """
    loading_matrices = [None for _ in tensor_slices]
    score_matrices = [None for _ in tensor_slices]

    _, n_cols = T.shape(tensor_slices[0])

    if max_rank is not None:
        rank_limit = min(n_cols, max_rank)
    else:
        rank_limit = n_cols

    for i, tensor_slice in enumerate(tensor_slices):
        n_rows, _ = T.shape(tensor_slice)

        if n_rows <= rank_limit and not compression_threshold:
            score_matrices[i] = tensor_slice
            continue

        U, s, Vh = svd_interface(tensor_slice, n_eigenvecs=rank_limit, method=svd)

        # Threshold SVD, keeping only singular values that satisfy s_i >= s_0 * epsilon
        # where epsilon is the compression threshold
        num_svds = len([s_i for s_i in s if s_i >= (s[0] * compression_threshold)])
        U, s, Vh = U[:, :num_svds], s[:num_svds], Vh[:num_svds, :]

        # Array broadcasting happens at the last dimension, since Vh is num_svds x n_cols
        # we need to transpose it, multiply in the singular values and then transpose
        # it again. This is equivalent to writing diag(s) @ Vh. If we skip the
        # transposes, we would get Vh @ diag(s), which is wrong.
        score_matrices[i] = T.transpose(s * T.transpose(Vh))
        loading_matrices[i] = U

    return score_matrices, loading_matrices


def svd_decompress_parafac2_tensor(parafac2_tensor, loading_matrices):
    """Decompress the factors obtained by fitting PARAFAC2 on SVD-compressed data

    Decompress a PARAFAC2 decomposition that describes the compressed data so that it
    models the original uncompressed data. Fitting to compressed data, and then
    decompressing is mathematically equivalent to fitting to the uncompressed data.

    See :py:meth:`svd_compress_tensor_slices` for information about SVD-compression and
    decompression.

    .. note::
        To decompress the data, we left-multiply the loading-matrices into the
        :math:`B_i`-matrices. However, :math:`B_i = P_i B`, so the decompression is
        implemented by left-multiplying the loading matrices by the :math:`P_i`-matrices.

    Parameters
    ----------
    parafac2_tensor: tl.Parafac2Tensor
        A decomposition obtained from fitting a PARAFAC2 model to compressed data
    loading_matrices: list of matrices
        Loading matrices obtained when compressing the data. See
        :py:meth:`svd_compress_tensor_slices` for more information.

    Returns
    -------
    tl.Parafac2Tensor:
        Decompressed PARAFAC2 decomposition - equivalent to the decomposition we would
        get from fitting parafac2 to uncompressed data.
    """
    weights, factors, projections = parafac2_tensor
    projections = projections.copy()

    for i, projection in enumerate(projections):
        if loading_matrices[i] is not None:
            projections[i] = T.matmul(loading_matrices[i], projection)

    return Parafac2Tensor((weights, factors, projections))
