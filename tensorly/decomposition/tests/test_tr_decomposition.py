from .._tr import tensor_ring
from ...random import random_tr
from ...testing import assert_, assert_array_almost_equal, assert_raises


def test_tensor_ring():
    """ Test for tensor_ring """
    # Create tensor with random elements
    tensor_shape = (6, 2, 3, 2, 6)
    rank = (3, 2, 4, 12, 18, 3)
    tensor = random_tr(tensor_shape, rank, full=True, random_state=1234)

    # Compute TR decomposition
    tr_tensor = tensor_ring(tensor, rank)
    assert_(len(tr_tensor.factors) == len(tensor_shape),
            f"Number of factors should be {len(tensor_shape)}, currently has {len(tr_tensor.factors)}")

    for k in range(len(tensor_shape)):
        (r_prev_k, n_k, r_k) = tr_tensor[k].shape
        assert_(n_k == tensor_shape[k], f"Mode 2 of factor {k} should have {tensor_shape[k]} dimensions, currently has {n_k}")
        assert_(r_prev_k == rank[k], "Incorrect ranks")
        if k:
            assert_(r_prev_k == r_prev_iteration, "Incorrect ranks")
        r_prev_iteration = r_k

    assert_array_almost_equal(tr_tensor.to_tensor(), tensor, decimal=2)


def test_tensor_ring_mode():
    """ Test for tensor_ring `mode` argument"""
    # Create tensor with random elements
    tensor_shape = (6, 2, 3, 2, 6)
    rank = (12, 2, 1, 3, 6, 12)
    tensor = random_tr(tensor_shape, rank, full=True, random_state=1234)

    # Compute TR decomposition
    tr_tensor = tensor_ring(tensor, rank, mode=1)
    assert_(len(tr_tensor.factors) == len(tensor_shape),
            f"Number of factors should be {len(tensor_shape)}, currently has {len(tr_tensor.factors)}")

    for k in range(len(tensor_shape)):
        (r_prev_k, n_k, r_k) = tr_tensor[k].shape
        assert_(n_k == tensor_shape[k], f"Mode 2 of factor {k} should have {tensor_shape[k]} dimensions, currently has {n_k}")
        assert_(r_prev_k == rank[k], "Incorrect ranks")
        if k:
            assert_(r_prev_k == r_prev_iteration, "Incorrect ranks")
        r_prev_iteration = r_k

    assert_array_almost_equal(tr_tensor.to_tensor(), tensor, decimal=2)

    with assert_raises(ValueError):
        tensor_ring(tensor, rank=(12, 2, 10, 3, 6, 12), mode=1)
