import numpy as np

import tensorly as tl
from tensorly.testing import assert_equal
from ..similarity import calc_corrindex


def test_calc_corrindex():
    """Test column permutation invariance and scaling invariance of calc_corrindex"""

    # initialize random matrix
    n = np.random.randint(2,100) # matrix shapes
    X_1 = tl.random.random_tensor((n, n))

    X_perm = X_1[:,np.random.permutation(n)] # create a column permutation

    scalar = 0
    while scalar == 0:
        scalar = np.random.randint(-100,100) # ensures diagonal is not 0
    X_scaled = tl.matmul(X_1, tl.diag([scalar]*n)) # create a scaled matrix

    # processing to get C as input
    X_1 = X_1/tl.norm(X_1, axis = 0)
    X_perm = X_perm/tl.norm(X_perm, axis = 0)
    X_scaled = X_scaled/tl.norm(X_scaled, axis = 0)

    C_perm = tl.abs(tl.matmul(tl.conj(X_1.T), X_perm))
    C_scaled = tl.abs(tl.matmul(tl.conj(X_1.T), X_scaled))

    tol = 5e-15

    # test column permutation invariance
    score_perm = calc_corrindex(C_perm)
    if score_perm < tol:
        score_perm = 0
    assert_equal(score_perm, 0)

    # test scaling invariance
    score_scaled = calc_corrindex(C_scaled)
    if score_scaled < tol:
        score_scaled = 0
    assert_equal(score_scaled, 0)
