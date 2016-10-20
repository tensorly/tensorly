import numpy as np
from scipy.linalg import norm as sc_norm
from numpy.testing import assert_
from ...base import tensor_to_vec
from .._norm import norm

# Author: Jean Kossaifi


def test_norm():
    """Test for norm"""
    tensor = np.array([[[1, 2],
                       [0.5, 0.5]],
                      [[-3, -0.5],
                       [0.5, -1]]])
    true_res_order2 = 4
    true_res_order1 = 9
    res_order2 = norm(tensor, order=2)
    res_order1 = norm(tensor, order=1)
    assert_(true_res_order1 == res_order1)
    assert_(true_res_order2 == res_order2)
    assert_(norm(tensor, 0.5) == sc_norm(tensor_to_vec(tensor), 0.5))
