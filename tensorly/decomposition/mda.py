import numpy as np
from .. import backend as T
from ..base import unfold
from ..tenalg import multi_mode_dot


# Author: James Oldfield

# License: BSD 3 clause


def mda(X, y, ranks, n_iters=5):
    """Multilinear Linear Discriminant Analysis (MDA).

        Learns a projection matrix for each mode of the data tensor
        to project the input tensor into a low-dimensional tensor subspace
        where the *scatter ratio criterion* is maximised.

    Parameters
    ----------
    X : ndarray
        tensor data of shape (n_samples, N1, ..., NS). Note: the first dimension is the sample dimension.
    y : ndarray
        list of length (n_samples) containing the integer-valued class label of tensor sample i.
    ranks : list
        list of integers determining the dimension of the subspace for mode-k.
    n_iters : int, optional, default is 5
        number of steps to employ the APP scheme for.

    Returns
    -------
    factors : list
        list of the learnt projection matrices for each mode

    Notes
    -----

    This implementation computes the *Constrained Multilinear Discriminant Analysis* (CMDA) solution as presented in [1].

    Given the learnt factor matrices, one can then compute the projection along all modes with::

      factors = mda(X_train, y_train, ranks, n)
      Z = tl.tenalg.multi_mode_dot(X_train, factors, modes=[1, 2], transpose=True)


    The so-called **scatter ratio objective function** (a higher-order extension of the Fisher criterion) for each mode n is maximised. Concretely, both the mode-n between- and within-class scatter matrices are computed. First a global mean tensor :math:`\\mathcal{M}` (the higher-order analogue of the mean vector :math:`\\mathbf{m}` in the standard LDA setting), and a set of class-specific mean tensors :math:`\\mathcal{M}_i` for :math:`i=1,\\dots, c` are computed. The class-specific mean tensors are computed as

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            \\mathcal{M}_i= \\frac{1}{n_i} \\sum_{j=1}^{n_i} \\mathcal{X}_{ij},
        \\end{aligned}
        \\end{equation*}

    and the global mean tensor is computed as

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            \\mathcal{M}= \\frac{1}{N} \\sum_{i=1}^{c} \\sum_{j=1}^{n_i} \\mathcal{X}_{ij}.
        \\end{aligned}
        \\end{equation*}

    Following this, the mode-n between- and within-class scatter matrices are computed (following [1]) as:

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            \\mathbf{B}_n^{\\bar{n}}
                = \\sum_{i=1}^{c} n_i \\left[ \\left(\\mathcal{M}_i - \\mathcal{M}\\right) \\prod_{\\substack{k=1 \\\\ k\\neq n}}^{N} \\times_k {\\mathbf{U}^{(k)}}^\\top \\right]_{[n]}
                \\left[ \\left(\\mathcal{M}_i - \\mathcal{M}\\right) \\prod_{\\substack{k=1 \\\\ k\\neq n}}^{N} \\times_k {\\mathbf{U}^{(k)}}^\\top \\right]_{[n]}^\\top,
        \\end{aligned}
        \\end{equation*}

    for the **between-class scatter**, and

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            \\mathbf{W}_n^{\\bar{n}}
                = \\sum_{i=1}^{c} \\sum_{j=1}^{n_i} \\left[ \\left(\\mathcal{X}_{ij} - \\mathcal{M}_i\\right) \\prod_{\\substack{k=1 \\\\ k\\neq n}}^{N} \\times_k {\\mathbf{U}^{(k)}}^\\top \\right]_{[n]}
                \\left[ \\left(\\mathcal{X}_{ij} - \\mathcal{M}_i\\right) \\prod_{\\substack{k=1 \\\\ k\\neq n}}^{N} \\times_k {\\mathbf{U}^{(k)}}^\\top \\right]_{[n]}^\\top,
        \\end{aligned}
        \\end{equation*}

    for the **within-class scatter**, with :math:`\\mathbf{X}_{[n]}` denoting the mode-n unfolding of tensor :math:`\\mathcal{X}`.

    - [1] Q. Li and D. Schonfeld, "Multilinear Discriminant Analysis for Higher-Order Tensor Data Classification," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 12, pp. 2524-2537, 1 Dec. 2014, doi: 10.1109/TPAMI.2014.2342214.

    """

    ###############
    # check correct # of ranks have been supplied
    ###############
    assert len(ranks) == len(T.shape(X)[1:]), 'Expected number of ranks: {}. \
        But number supplied is {}'.format(len(T.shape(X)[1:]), len(ranks))

    backend = T.get_backend()

    global_mean = T.mean(X, axis=0)
    class_means = []

    # ith element will contain a list of the indices of training data with class label i
    class_idx = [[] for _ in range(len(set(y)))]

    # store the training data's class label at class index
    for i, label in enumerate(y):
        class_idx[label] += [i]

    # store the mean of all tensor samples with label i
    for i in range(len(set(y))):
        # tensorflow is only backend to not support indexing into tensor with a list
        if backend == 'tensorflow':
            class_means += [T.mean(T.tensor([X[j] for j in class_idx[i]]), axis=0)]
        else:
            class_means += [T.mean(X[class_idx[i], ...], axis=0)]

    # the first mode is the 'sample' mode
    num_modes = len(T.shape(X)) - 1

    # initialise the factor matrices as 1-matrices
    factors = [T.ones((dim, T.shape(X)[i + 1]), **T.context(X))
               for i, dim in enumerate(list(T.shape(X))[1:])]

    for t in range(1, n_iters + 1):
        # for each iteration compute partial projections for mode k,
        # i.e. project along all modes but k.
        for k in range(num_modes):
            B_scat, W_scat = compute_modek_wb_scatters(X, k, factors, global_mean, class_means, class_idx)

            # first compute the inverse of the scatter matrix
            # i.e. solve SX=I for X
            W_scat_inv = T.solve(W_scat, T.eye(T.shape(W_scat)[0]))

            ###################################################
            # *Constrained Multilinear Discriminant Analysis* (CMDA) [1] solution for factor matrix U_k is given by
            # top `rank' number of left-singular vectors of W^{_1}B.
            # --
            # [1] Q. Li et al. "Multilinear Discriminant Analysis for Higher-Order Tensor Data Classification"
            ###################################################
            U, _, _ = T.partial_svd(T.dot(W_scat_inv,  B_scat))
            factors[k] = U[:, :ranks[k]]

    return factors


def compute_modek_wb_scatters(X, mode, factors, global_mean, class_means, class_idx):
    """Computes the mode-k between- and within-class scatter matrices in the partially projected tensor subspace.

    Parameters
    ----------
    X : ndarray
        tensor data of shape (n_samples, N1, ..., NS)
    mode : int
        desired mode to compute mode-n scatter matrices for
    global_mean : ndarray
        global mean tensor of shape (N1, ..., NS)
    class_means : list
        list of mean tensors for each class. Each element is of shape (N1, ..., NS)
    class_idx : list
        list of indices of the training examples belonging to class label i

    Returns
    -------
    B_scat : ndarray
        the mode-n between-class matrix (matrix)
    W_scat : ndarray
        the mode-n within-class matrix (matrix)

    Notes
    -----
    For the computation of the mode-n between- and within-class scatter matrices, first a global mean tensor :math:`\\mathcal{M}` (the higher-order analogue of the mean vector :math:`\\mathbf{m}` in the standard LDA setting), and a set of class-specific mean tensors :math:`\\mathcal{M}_i` for :math:`i=1,\\dots, c` are computed. The class-specific mean tensors are computed as

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            \\mathcal{M}_i= \\frac{1}{n_i} \\sum_{j=1}^{n_i} \\mathcal{X}_{ij},
        \\end{aligned}
        \\end{equation*}

    and the global mean tensor is computed as

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            \\mathcal{M}= \\frac{1}{N} \\sum_{i=1}^{c} \\sum_{j=1}^{n_i} \\mathcal{X}_{ij}.
        \\end{aligned}
        \\end{equation*}

    Following this, the mode-n between- and within-class scatter matrices are computed (following [1]) as:

    .. math::
       :nowrap:

        \\begin{equation*}
        \\begin{aligned}
            \\mathbf{B}_n^{\\bar{n}}
                = \\sum_{i=1}^{c} n_i \\left[ \\left(\\mathcal{M}_i - \\mathcal{M}\\right) \\prod_{\\substack{k=1 \\\\ k\\neq n}}^{N} \\times_k {\\mathbf{U}^{(k)}}^\\top \\right]_{[n]}
                \\left[ \\left(\\mathcal{M}_i - \\mathcal{M}\\right) \\prod_{\\substack{k=1 \\\\ k\\neq n}}^{N} \\times_k {\\mathbf{U}^{(k)}}^\\top \\right]_{[n]}^\\top,
        \\end{aligned}
        \\end{equation*}

    - [1] Q. Li and D. Schonfeld, "Multilinear Discriminant Analysis for Higher-Order Tensor Data Classification," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, no. 12, pp. 2524-2537, 1 Dec. 2014, doi: 10.1109/TPAMI.2014.2342214.

    """
    B_scat = 0
    W_scat = 0

    num_classes = len(class_means)
    num_each_class = [len(c) for c in class_idx]

    # outer loop is over each class label, to build the between-scatter matrices
    for c in range(num_classes):
        M = class_means[c] - global_mean
        proj_but_k = unfold(multi_mode_dot(M, factors, transpose=True, skip=mode), mode)
        B_scat += num_each_class[c] * T.dot(proj_but_k, T.transpose(proj_but_k))

        # inner loop for within-class computation
        for j in range(num_each_class[c]):
            # subtract mean for class c from jth sample of class c
            M = X[class_idx[c][j]] - class_means[c]

            proj_but_k = unfold(multi_mode_dot(M, factors, transpose=True, skip=mode), mode)
            W_scat += T.dot(proj_but_k, T.transpose(proj_but_k))

    return B_scat, W_scat
