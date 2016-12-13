from ..base import cp_tensor, tucker_tensor
from ...tucker import tucker_to_tensor
from ...tenalg import multi_mode_dot
from ...base import unfold
from numpy.linalg import matrix_rank 
from numpy.testing import assert_equal, assert_array_almost_equal, assert_raises
import numpy as np


def test_cp_tensor():
    """test for random.cp_tensor"""
    shape = (10, 11, 12)
    rank = 4
    
    tensor = cp_tensor(shape, rank, full=True)
    for i in range(tensor.ndim):
        assert_equal(matrix_rank(unfold(tensor, i)), rank)
        
    factors = cp_tensor(shape, rank, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))
        
def test_tucker_tensor():
    """test for random.tucker_tensor"""
    shape = (10, 11, 12)
    rank = 4
    
    tensor = tucker_tensor(shape, rank, full=True)
    for i in range(tensor.ndim):
        assert_equal(matrix_rank(unfold(tensor, i)), rank)
        
    core, factors = tucker_tensor(shape, rank, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))
    
    shape = (10, 11, 12)
    rank = (6, 4, 5)
    tensor = tucker_tensor(shape, rank, full=True)
    for i in range(tensor.ndim):
        assert_equal(matrix_rank(unfold(tensor, i)),  min(shape[i], rank[i]))
        
    core, factors = tucker_tensor(shape, rank, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank[i]),
                err_msg=('{}-th factor has shape {}, expected {}.'.format(
                     i, factor.shape, (shape[i], rank[i]))))
    assert_equal(core.shape, rank, err_msg='core has shape {}, expected {}.'.format(
                                     core.shape, rank))
    for factor in factors:
        assert_array_almost_equal(factor.T.dot(factor), np.eye(factor.shape[1]))
    tensor = tucker_to_tensor(core, factors)
    reconstructed = multi_mode_dot(tensor, factors, transpose=True)
    assert_array_almost_equal(core, reconstructed)

    with assert_raises(ValueError):
        tucker_tensor((3, 4, 5), (3, 6, 3))
