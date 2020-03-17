import numpy as np

from .. import backend as tl
from ..base import unfold, tensor_to_vec
from ..tucker_tensor import (tucker_to_tensor, tucker_to_unfolded,
                             tucker_to_vec, _validate_tucker_tensor,
                             tucker_mode_dot)
from ..tenalg import kronecker, mode_dot
from ..testing import (assert_array_equal, assert_array_almost_equal, 
                       assert_equal, assert_raises)
from ..random import check_random_state, random_tucker


def test_validate_tucker_tensor():
    rng = check_random_state(12345)
    true_shape = (3, 4, 5)
    true_rank = (3, 2, 4)
    core, factors = random_tucker(true_shape, rank=true_rank)
    
    # Check shape and rank returned
    shape, rank = _validate_tucker_tensor((core, factors))
    assert_equal(shape, true_shape,
                    err_msg='Returned incorrect shape (got {}, expected {})'.format(
                        shape, true_shape))
    assert_equal(rank, true_rank,
                    err_msg='Returned incorrect rank (got {}, expected {})'.format(
                        rank, true_rank))

    # One of the factors has the wrong rank
    factors[0], copy = tl.tensor(rng.random_sample((4, 4))), factors[0]
    with assert_raises(ValueError):
        _validate_tucker_tensor((core, factors))
    
    # Not enough factors to match core
    factors[0] = copy
    with assert_raises(ValueError):
        _validate_tucker_tensor((core, factors[1:]))

    # Not enough factors
    with assert_raises(ValueError):
        _validate_tucker_tensor((core, factors[:1]))


def test_tucker_to_tensor():
    """Test for tucker_to_tensor"""
    X = tl.tensor([[[1, 13],
                   [4, 16],
                   [7, 19],
                   [10, 22]],

                  [[2, 14],
                   [5, 17],
                   [8, 20],
                   [11, 23]],

                  [[3, 15],
                   [6, 18],
                   [9, 21],
                   [12, 24]]])
    ranks = [2, 3, 4]
    U = [tl.tensor(np.arange(R * s).reshape((R, s))) for (R, s) in zip(ranks, tl.shape(X))]
    true_res = np.array([[[390, 1518, 2646, 3774],
                         [1310, 4966, 8622, 12278],
                         [2230, 8414, 14598, 20782]],
                        [[1524, 5892, 10260, 14628],
                         [5108, 19204, 33300, 47396],
                         [8692, 32516, 56340, 80164]]])
    res = tucker_to_tensor((X, U))
    assert_array_equal(true_res, res)


def test_tucker_to_unfolded():
    """Test for tucker_to_unfolded

    Notes
    -----
    Assumes that tucker_to_tensor is properly tested
    """
    G = tl.tensor(np.random.random((4, 3, 5, 2)))
    ranks = [2, 2, 3, 4]
    U = [tl.tensor(np.random.random((ranks[i], G.shape[i]))) for i in range(tl.ndim(G))]
    full_tensor = tucker_to_tensor((G, U))
    for mode in range(tl.ndim(G)):
        assert_array_almost_equal(tucker_to_unfolded((G, U), mode), unfold(full_tensor, mode))
        assert_array_almost_equal(tucker_to_unfolded((G, U), mode),
                                    tl.dot(tl.dot(U[mode], unfold(G, mode)), tl.transpose(kronecker(U, skip_matrix=mode))),
                                    decimal=5)


def test_tucker_to_vec():
    """Test for tucker_to_vec

    Notes
    -----
    Assumes that tucker_to_tensor works correctly
    """
    G = tl.tensor(np.random.random((4, 3, 5, 2)))
    ranks = [2, 2, 3, 4]
    U = [tl.tensor(np.random.random((ranks[i], G.shape[i]))) for i in range(tl.ndim(G))]
    vec = tensor_to_vec(tucker_to_tensor((G, U)))
    assert_array_almost_equal(tucker_to_vec((G, U)), vec)
    assert_array_almost_equal(tucker_to_vec((G, U)), tl.dot(kronecker(U), tensor_to_vec(G)), decimal=5)


def test_tucker_mode_dot():
    """Test for tucker_mode_dot
    
        We will compare tucker_mode_dot 
        (which operates directly on decomposed tensors)
        with mode_dot (which operates on full tensors)
        and check that the results are the same.
    """
    rng = check_random_state(12345)
    shape = (5, 4, 6)
    rank = (3, 2, 4)
    tucker_ten = random_tucker(shape, rank=rank, full=False, random_state=rng)
    full_tensor = tucker_to_tensor(tucker_ten)
    # matrix for mode 1
    matrix = tl.tensor(rng.random_sample((7, shape[1])))
    # vec for mode 2
    vec = tl.tensor(rng.random_sample(shape[2]))

    # Test tucker_mode_dot with matrix
    res = tucker_mode_dot(tucker_ten, matrix, mode=1, copy=True)
    # Note that if copy=True is not respected, factors will be changes
    # And the next test will fail
    res = tucker_to_tensor(res)
    true_res = mode_dot(full_tensor, matrix, mode=1)
    assert_array_almost_equal(true_res, res, decimal=5)
    
    # Check that the data was indeed copied
    rec = tucker_to_tensor(tucker_ten)
    assert_array_almost_equal(full_tensor, rec, decimal=5)
    
    # Test tucker_mode_dot with vec
    res = tucker_mode_dot(tucker_ten, vec, mode=2, copy=True)
    res = tucker_to_tensor(res)
    true_res = mode_dot(full_tensor, vec, mode=2)
    assert_equal(res.shape, true_res.shape)
    assert_array_almost_equal(true_res, res, decimal=5)