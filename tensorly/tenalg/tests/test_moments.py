import numpy as np

import tensorly as tl
from tensorly import testing

from .. import higher_order_moment


def test_higher_order_moment():
    """Test higher_order_moment against the mean of outer products"""
    rng = tl.check_random_state(1234)

    X = rng.random_sample((5, 3))
    testing.assert_array_almost_equal(
        higher_order_moment(tl.tensor(X), 1), np.mean(X, axis=0)
    )
    testing.assert_array_almost_equal(
        higher_order_moment(tl.tensor(X), 3), np.einsum("ni,nj,nk->ijk", X, X, X) / 5
    )

    # Samples that are themselves tensors
    Y = rng.random_sample((4, 2, 3))
    testing.assert_array_almost_equal(
        higher_order_moment(tl.tensor(Y), 2), np.einsum("nij,nkl->ijkl", Y, Y) / 4
    )
