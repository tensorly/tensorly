import numpy as np
from numpy.linalg import inv, eig

import tensorly as tl
from ..mpca import mpca, compute_modek_total_scatter
from ...testing import assert_array_almost_equal

from ...tenalg import multi_mode_dot


def test_mpca():
    """Test for MPCA

    (1) Compute the (square) projection matrices for each mode.
        Then check we recover the original data with the inverse projections.

    (2) Test the numerical equivalence of the np eigendecomp and tl.partial_svd
        for (positive definite) scatter matrices.

    """
    tol = 1e-3

    np.random.seed(1234)

    # 10 random 3rd-order tensors
    X = tl.tensor(np.random.randn(10, 5, 5, 5))

    ###########################################
    # (1) Check reconstruction of original data
    ###########################################
    factors = mpca(X, ranks=[5, 5, 5], n_iters=5)

    # project onto MPCA tensor subspace
    Z = multi_mode_dot(X, factors, modes=[1, 2, 3], transpose=True)

    # recover X, using the inverse projection matrices
    X_hat = multi_mode_dot(Z, [tl.tensor(inv(tl.to_numpy(f))) for f in factors], modes=[1, 2, 3], transpose=True)

    assert_array_almost_equal(X, X_hat, decimal=tol)

    #####################################################
    # (2) Check left-singular vector / eigenvector equiv.
    #     for scatter matrix (up to a sign)
    #####################################################
    scatter = compute_modek_total_scatter(X, mode=0, factors=factors)

    _, U_eig = eig(tl.to_numpy(scatter))
    U_svd, _, _ = tl.partial_svd(scatter)

    assert_array_almost_equal(
        tl.abs(tl.tensor(U_eig)),
        tl.abs(U_svd),
        decimal=tol)
