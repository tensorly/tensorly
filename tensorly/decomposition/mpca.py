import numpy as np
from .. import backend as T
from ..base import unfold
from ..tenalg import multi_mode_dot


# Author: James Oldfield

# License: BSD 3 clause


def mpca(X, ranks, n_iters=5, zero_mean=True):
    """Multilinear Principal Component Analysis, via Alternating Partial Projections.

        Learns a projection matrix for each mode of the data tensor
        to project the input tensor into a low-dimensional tensor subspace
        where the total scatter is maximised.

        Note: the first dimension is the sample dimension.

    Parameters
    ----------
    X : ndarray
        tensor data of shape (n_samples, N1, ..., NS)
    ranks : list
        list of integers determining the dimension of the subspace for mode-k.
    n_iters : int, optional, default is 5
        number of steps to employ the APP scheme for.
    zero_mean : bool, optional, default is True
        The sample-wise mean needs to be removed (along 0th axis).
        Pass `zero_mean=False` if this has already been removed.

    Returns
    -------
    factors : list
        list of the learnt projection matrices for each mode

    Notes
    -----
    In a similar manner to regular PCA, *Multilinear PCA* (MPCA) seeks a projection onto a (tensor) subspace where the total scatter is maximised [1]. In contrast to PCA, MPCA seeks not a single projection matrix, but :math:`k` projection matrices :math:`\\mathbf{U}^{(k)}` for each mode :math:`k` of a higher-order tensor, :math:`\\mathcal{X}\\in\\mathbb{R}^{I_1\\times I_2\\times\\cdots\\times I_{N}}`.

    The total scatter maximisation objective defined for the projection matrices as

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            J(\\mathbf{U}^{(k)}) = \\underset{\\mathbf{U}^{(k)}}{\\arg\\max}
                \\sum_{m=1}^M || \\tilde{\\mathcal{X}}_m \\prod_{n=1}^N \\times_n {\\mathbf{U}^{(n)}}^\\top ||^2_F,
        \\end{aligned}
        \\end{equation*}

    where :math:`\\tilde{\mathcal{X}}_m` is the :math:`m` th data point with the sample mean :math:`\\bar{\\mathcal{X}} = \\frac{1}{M} \\sum_{m=1}^{M}\\mathcal{X}_m` subtracted.

    For example, for a tensor of 4th order, one can project each sample onto its low-dimensional tensor subspace using::

        tl.tenalg.multi_mode_dot(X, factors, modes=[1, 2, 3], transpose=True)

    - [1] Haiping Lu, K. N. Plataniotis, and A. N. Venetsanopoulos, ‘MPCA: Multilinear Principal Component Analysis of Tensor Objects’, IEEE Trans. Neural Netw., vol. 19, no. 1, pp. 18–39, Jan. 2008, doi: 10.1109/TNN.2007.901277.

    """

    ###############
    # check correct # of ranks have been supplied
    ###############
    assert len(ranks) == len(X.shape[1:]), 'Expected number of ranks: {}. \
        But number supplied is {}'.format(len(X.shape[1:]), len(ranks))

    if zero_mean:
        # first, zero-mean the tensor data
        X = X - T.mean(X, axis=0)

    # the first mode is the 'sample' mode
    num_modes = len(X.shape) - 1

    # initialise the factor matrices as 1-matrices
    factors = [T.ones((dim, X.shape[i + 1]), **T.context(X))
               for i, dim in enumerate(list(X.shape)[1:])]

    for t in range(1, n_iters + 1):
        # for each iteration compute partial projections for mode k,
        # i.e. project along all modes but k.
        for k in range(num_modes):
            scatter = compute_modek_total_scatter(X, k, factors)

            # set the factor matrices equal to the top-`ranks[k]` number of eigenvectors
            # note: scatter is a positive definite matrix, thus the left-singular vectors
            # are the same as its eigenvectors (up to a sign).
            U, _, _ = T.partial_svd(scatter)
            factors[k] = U[:, :ranks[k]]

    return factors


def compute_modek_total_scatter(X, mode, factors):
    """Computes the mode-n total scatter matrix for the partial projections of :math:`\\mathcal{X}_m` along all modes but :math:`n`.

    Parameters
    ----------
    X : ndarray
        tensor data of shape (n_samples, N1, ..., NS)
    mode : int
        desired mode to compute mode-n total scatter for.
    factors : list
        list of tensorly tensors (matrices) containing the projection matrices.

    Returns
    -------
    scatter : ndarray
        the mode-n total scatter (matrix)

    Notes
    -----
    Following [1], this is defined as

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
        \\mathbf{S}^{(n)}_{T_{\\hat{\\mathcal{Y}}}} =
            \\sum_{m=1}^{M}
                \\left( \\mathbf{\\hat{Y}}_{m[n]} - \\mathbf{\\bar{\\hat{Y}}}_{[n]} \\right)
                \\left( \\mathbf{\\hat{Y}}_{m[n]} - \\mathbf{\\bar{\\hat{Y}}}_{[n]} \\right)^\\top,
        \\end{aligned}
        \\end{equation*}

    where :math:`\\bar{\\hat{\\mathbf{Y}}}_{[n]}` is the mode-n unfolding of the projection of the mean tensor :math:`\\bar{\\mathcal{X}}` along all modes but :math:`n`.

    - [1] Multilinear Subspace Learning: Dimensionality Reduction of Multidimensional Data, Haiping Lu, K. N. Plataniotis, and A. N. Venetsanopoulos, Chapman & Hall/CRC Press Machine Learning and Pattern Recognition Series, Taylor and Francis, ISBN: 978-1-4398572-4-3, December 2013.

    """
    scatter = 0

    ###############
    # check that all factor matrices have been supplied
    ###############
    assert len(factors) == len(X.shape[1:]), 'Expected number of factor matrices: {}. \
        But number found is {}'.format(len(X.shape[1:]), len(factors))

    ###############
    # check that dimensions of factor matrices are compatible
    ###############
    for i in range(len(X.shape) - 1):
        assert X.shape[i + 1] == factors[i].shape[0], 'Incompatible dimensions for factor matrix {}. \
            U_k must be of size ({}, P_k), but is of size ({}, P_k)'.format(i, X.shape[i + 1], factors[i].shape[0])

    # loop over each data point, building the mode-n total scatter matrix
    for m in range(len(X)):
        proj_but_k = unfold(multi_mode_dot(X[m], factors, transpose=True, skip=mode), mode)
        scatter += T.dot(proj_but_k, T.transpose(proj_but_k))

    return scatter
