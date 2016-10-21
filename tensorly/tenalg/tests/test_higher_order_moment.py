import numpy as np
from numpy.testing import assert_array_almost_equal
from ..higher_order_moment import higher_order_moment


def test_higher_order_moment():
    """Test for higher_order_moment"""
    X = np.array([[1, 0],[0, 1], [1, 1], [-1, 1], [1, -1.5]])
    centered_X = X - X.mean(axis=0)
    order_2 = centered_X.T.dot(centered_X)/X.shape[0]
    order_3 = np.array([[[-0.432,  0.196],
                         [ 0.196,  0.282]],
                        [[ 0.196,  0.282],
                         [ 0.282, -0.966]]])
    order_4 = np.array([[[[ 0.8512, -0.4536],
                          [-0.4536,  0.4828]],
                         [[-0.4536,  0.4828],
                          [ 0.4828, -0.7854]]],
                        [[[-0.4536,  0.4828],
                          [ 0.4828, -0.7854]],
                         [[ 0.4828, -0.7854],
                          [-0.7854,  2.2452]]]])
    assert_array_almost_equal(higher_order_moment(X, 2), order_2)
    assert_array_almost_equal(higher_order_moment(X, 3), order_3)
    assert_array_almost_equal(higher_order_moment(X, 4), order_4)
