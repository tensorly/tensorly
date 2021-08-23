import numpy as np
from numpy.linalg import matrix_rank

from ... import backend as T
from ..base import random_cp, random_tucker, random_tt, random_tr
from ...tucker_tensor import tucker_to_tensor
from ...tenalg import multi_mode_dot
from ...base import unfold
from ...testing import assert_equal, assert_array_almost_equal, assert_raises


def test_check_random_state():
    """Test for check_random_state"""

    # Generate a random state for me
    rns = T.check_random_state(seed=None)
    assert(isinstance(rns, np.random.RandomState))

    # random state from integer seed
    rns = T.check_random_state(seed=10)
    assert(isinstance(rns, np.random.RandomState))

    # if it is already a random state, just return it
    cpy_rns = T.check_random_state(seed=rns)
    assert(cpy_rns is rns)

    # only takes as seed a random state, an int or None
    assert_raises(ValueError, T.check_random_state, seed='bs')

def test_random_cp():
    """test for random.random_cp"""
    shape = (10, 11, 12)
    rank = 4

    tensor = random_cp(shape, rank, full=True)
    assert T.shape(tensor) == shape

    for i in range(T.ndim(tensor)):
        assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i)), tol=1e-6), rank)

    weights, factors = random_cp(shape, rank, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))

    # tests that the columns of each factor matrix are indeed orthogonal
    weights, factors = random_cp(shape, rank, full=False, orthogonal=True)
    for i, factor in enumerate(factors):
        for j in range(rank):
            for k in range(j):
                # (See issue #40)
                dot_product = T.dot(factor[:,j], factor[:,k])
                try:
                    T.shape(dot_product)
                except:
                    dot_product = T.tensor([dot_product], **T.context(weights))
                assert_array_almost_equal(dot_product, T.tensor([0]))

    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = 11
        _ = random_cp(shape, rank, orthogonal=True)

def test_random_tucker():
    """test for random.random_tucker"""
    shape = (10, 11, 12)
    rank = 4

    tensor = random_tucker(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))), rank)

    core, factors = random_tucker(shape, rank, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))

    shape = (10, 11, 12)
    rank = (6, 4, 5)
    tensor = random_tucker(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))),  min(shape[i], rank[i]))

    core, factors = random_tucker(shape, rank, orthogonal=True, full=False)
    for i, factor in enumerate(factors):
        assert_equal(factor.shape, (shape[i], rank[i]),
                err_msg=('{}-th factor has shape {}, expected {}.'.format(
                     i, factor.shape, (shape[i], rank[i]))))
    assert_equal(core.shape, rank, err_msg='core has shape {}, expected {}.'.format(
                                     core.shape, rank))
    for factor in factors:
        assert_array_almost_equal(T.dot(T.transpose(factor), factor),
                                  T.tensor(np.eye(factor.shape[1])))
    tensor = tucker_to_tensor((core, factors))
    reconstructed = multi_mode_dot(tensor, factors, transpose=True)
    assert_array_almost_equal(core, reconstructed)


def test_random_tt():
    """test for random.random_tt"""
    shape = (10, 11, 12)
    rank = (1, 4, 3, 1)
    true_shapes = [(1, 10, 4), (4, 11, 3), (3, 12, 1)]

    factors = random_tt(shape, rank, full=False)
    for i, (true_shape, factor) in enumerate(zip(true_shapes, factors)):
        assert_equal(factor.shape, true_shape,
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, true_shape)))

    # Missing a rank
    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = (1, 3, 1)
        _ = random_tt(shape, rank)

    # Not respecting the boundary rank conditions
    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = (1, 3, 3, 3)
        _ = random_tt(shape, rank)


def test_random_tr():
    """test for random.random_tr"""
    shape = (10, 11, 12)
    rank = (2, 4, 3, 2)
    true_shapes = [(2, 10, 4), (4, 11, 3), (3, 12, 2)]

    factors = random_tr(shape, rank, full=False)
    for i, (true_shape, factor) in enumerate(zip(true_shapes, factors)):
        assert_equal(factor.shape, true_shape,
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, true_shape)))

    # Missing a rank
    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = (2, 4, 2)
        _ = random_tr(shape, rank)

    # Not respecting the boundary rank conditions
    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = (2, 3, 3, 3)
        _ = random_tr(shape, rank)
