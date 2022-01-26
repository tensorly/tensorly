import numpy as np

import tensorly as tl
from tensorly.testing import assert_equal
from ..similarity import correlation_index


def test_correlation_index():
    """Test column permutation invariance and scaling invariance of correlation_index"""

    # initialize random matrix
    n = np.random.randint(20,100) # possible matrix shapes
    X_1 = tl.random.random_tensor((n, n))

    X_perm = X_1[:,np.random.permutation(n)] # create a column permutation

    scalar = 0
    while scalar == 0:
        scalar = np.random.randint(-100,100) # ensures diagonal is not 0
    X_scaled = tl.matmul(X_1, tl.diag([scalar]*n)) # create a scaled matrix

    # randomly create intervals to separate matrix into factors list
    intervals = tl.tensor((1,1))
    while tl.any(np.diff(intervals) < 3):
        intervals = sorted(np.random.choice(range(3, n - 3), np.random.randint(3, 10), replace=False))
    intervals = [0] + intervals + [n]

    factors_1 = [X_1[intervals[i - 1]:intervals[i], :] for i in range(1, len(intervals))]
    factors_perm = [X_perm[intervals[i - 1]:intervals[i], :] for i in range(1, len(intervals))]
    factors_scaled = [X_scaled[intervals[i - 1]:intervals[i], :] for i in range(1, len(intervals))]

    # test column permutation invariance
    score_perm = correlation_index(factors_1, factors_perm)
    assert_equal(score_perm, 0)

    # test scaling invariance
    score_scaled = correlation_index(factors_1, factors_scaled)
    assert_equal(score_scaled, 0)
