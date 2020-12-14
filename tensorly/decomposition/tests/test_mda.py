import numpy as np
from numpy.linalg import inv

import tensorly as tl
from ..mda import mda
from ...testing import assert_array_almost_equal

from ...tenalg import multi_mode_dot


def test_mda():
    """Test for MDA

    (1) Compute the (square) projection matrices for each mode.
        Then check we recover the original data with the inverse projections.

    """
    tol = 1e-3

    np.random.seed(1234)

    # 10 random 3rd-order tensors
    X = tl.tensor(np.random.randn(10, 5, 5, 5))
    y = np.random.randint(0, 2, 10)

    ###########################################
    # (1) Check reconstruction of original data
    ###########################################
    factors = mda(X, y, ranks=[5, 5, 5], n_iters=5)

    # project onto MDA tensor subspace
    Z = multi_mode_dot(X, factors, modes=[1, 2, 3], transpose=True)

    # recover X, using the inverse projection matrices
    X_hat = multi_mode_dot(Z, [tl.tensor(inv(tl.to_numpy(f))) for f in factors], modes=[1, 2, 3], transpose=True)

    assert_array_almost_equal(X, X_hat, decimal=tol)
