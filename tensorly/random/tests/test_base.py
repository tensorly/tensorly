import numpy as np
from numpy.linalg import matrix_rank

import tensorly.backend as T
from tensorly.random.base import cp_tensor, tucker_tensor, check_random_state
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import multi_mode_dot
from tensorly.base import unfold
from tensorly.testing import assert_equal, assert_array_almost_equal, assert_raises


def test_check_random_state():
    """Test for check_random_state"""

    # Generate a random state for me
    rns = check_random_state(seed=None)
    assert(isinstance(rns, np.random.RandomState))

    # random state from integer seed
    rns = check_random_state(seed=10)
    assert(isinstance(rns, np.random.RandomState))

    # if it is already a random state, just return it
    cpy_rns = check_random_state(seed=rns)
    assert(cpy_rns is rns)

    # only takes as seed a random state, an int or None
    assert_raises(ValueError, check_random_state, seed='bs')


def test_cp_tensor():
    """test for random.cp_tensor"""
    shape = (10, 11, 12)
    rank = 4

    tensor = cp_tensor(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))), rank)

    factors = cp_tensor(shape, rank, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))

    # tests that the columns of each factor matrix are indeed orthogonal
    factors = cp_tensor(shape, rank, full=False, orthogonal=True)
    for i, factor in enumerate(factors):
        for j in range(rank):
            for k in range(j):
                # (See issue #40)
                dot_product = T.dot(factor[:,j], factor[:,k])
                try:
                    T.shape(dot_product)
                except:
                    dot_product = T.tensor([dot_product])
                assert_array_almost_equal(dot_product, T.tensor([0]))

    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = 11
        _ = cp_tensor(shape, rank, orthogonal=True)


def test_tucker_tensor():
    """test for random.tucker_tensor"""
    shape = (10, 11, 12)
    rank = 4

    tensor = tucker_tensor(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))), rank)

    core, factors = tucker_tensor(shape, rank, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))

    shape = (10, 11, 12)
    rank = (6, 4, 5)
    tensor = tucker_tensor(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))),  min(shape[i], rank[i]))

    core, factors = tucker_tensor(shape, rank, orthogonal=True, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank[i]),
                err_msg=('{}-th factor has shape {}, expected {}.'.format(
                     i, factor.shape, (shape[i], rank[i]))))
    assert_equal(core.shape, rank, err_msg='core has shape {}, expected {}.'.format(
                                     core.shape, rank))
    for factor in factors:
        assert_array_almost_equal(T.dot(T.transpose(factor), factor), T.tensor(np.eye(factor.shape[1])))
    tensor = tucker_to_tensor(core, factors)
    reconstructed = multi_mode_dot(tensor, factors, transpose=True)
    assert_array_almost_equal(core, reconstructed)

    with assert_raises(ValueError):
        tucker_tensor((3, 4, 5), (3, 6, 3))
