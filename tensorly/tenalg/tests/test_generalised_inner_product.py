from ... import backend as T
from ..generalised_inner_product import inner
import numpy as np

# Author: Jean Kossaifi
# License: BSD 3 clause

def test_inner():
    tensor_1 = T.tensor(np.arange(3*4).reshape((3, 4)))
    tensor_2 = T.tensor(np.arange(4*2).reshape((4, 2)))
    
    # For one common mode, equivalent to dot product
    res = inner(tensor_1, tensor_2, n_modes=1)
    true_res = T.dot(tensor_1, tensor_2)
    T.assert_array_almost_equal(res, true_res)
    
    # For no common mode, equivalent to inner product
    res = inner(tensor_1, tensor_1, n_modes=None)
    true_res = T.sum(tensor_1**2)
    T.assert_equal(res, true_res)
    
    # Inner product of tensors with different shapes is not defined
    with T.assert_raises(ValueError):
        inner(tensor_1, tensor_2)

