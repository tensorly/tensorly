from ..base import (random_kruskal, random_tucker,
                    random_mps, check_random_state)
from ...tucker_tensor import tucker_to_tensor
from ...tenalg import multi_mode_dot
from ...base import unfold
from numpy.linalg import matrix_rank
import numpy as np
from ... import backend as T


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
    T.assert_raises(ValueError, check_random_state, seed='bs')

def test_random_kruskal():
    """test for random.random_kruskal"""
    shape = (10, 11, 12)
    rank = 4

    tensor = random_kruskal(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        T.assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))), rank)

    factors = random_kruskal(shape, rank, full=False)
    for i, factor in enumerate(factors):
        T.assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))

    # tests that the columns of each factor matrix are indeed orthogonal
    factors = random_kruskal(shape, rank, full=False, orthogonal=True)
    for i, factor in enumerate(factors):
        for j in range(rank):
            for k in range(j):
                # (See issue #40)
                dot_product = T.dot(factor[:,j], factor[:,k])
                try:
                    T.shape(dot_product)
                except:
                    dot_product = T.tensor([dot_product])
                T.assert_array_almost_equal(dot_product, T.tensor([0]))

    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = 11
        _ = random_kruskal(shape, rank, orthogonal=True)

def test_random_tucker():
    """test for random.random_tucker"""
    shape = (10, 11, 12)
    rank = 4

    tensor = random_tucker(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        T.assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))), rank)

    core, factors = random_tucker(shape, rank, full=False)
    for i, factor in enumerate(factors):
        T.assert_equal(factor.shape, (shape[i], rank),
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, (shape[i], rank))))

    shape = (10, 11, 12)
    rank = (6, 4, 5)
    tensor = random_tucker(shape, rank, full=True)
    for i in range(T.ndim(tensor)):
        T.assert_equal(matrix_rank(T.to_numpy(unfold(tensor, i))),  min(shape[i], rank[i]))

    core, factors = random_tucker(shape, rank, orthogonal=True, full=False)
    for i, factor in enumerate(factors):
        T.assert_equal(factor.shape, (shape[i], rank[i]),
                err_msg=('{}-th factor has shape {}, expected {}.'.format(
                     i, factor.shape, (shape[i], rank[i]))))
    T.assert_equal(core.shape, rank, err_msg='core has shape {}, expected {}.'.format(
                                     core.shape, rank))
    for factor in factors:
        T.assert_array_almost_equal(T.dot(T.transpose(factor), factor), T.tensor(np.eye(factor.shape[1])))
    tensor = tucker_to_tensor(core, factors)
    reconstructed = multi_mode_dot(tensor, factors, transpose=True)
    T.assert_array_almost_equal(core, reconstructed)

    with T.assert_raises(ValueError):
        random_tucker((3, 4, 5), (3, 6, 3))


def test_random_mps():
    """test for random.random_mps"""
    shape = (10, 11, 12)
    rank = (1, 4, 3, 1)
    true_shapes = [(1, 10, 4), (4, 11, 3), (3, 12, 1)]

    factors = random_mps(shape, rank, full=False)
    for i, (true_shape, factor) in enumerate(zip(true_shapes, factors)):
        T.assert_equal(factor.shape, true_shape,
                err_msg=('{}-th factor has shape {}, expected {}'.format(
                     i, factor.shape, true_shape)))

    # Missing a rank 
    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = (1, 3, 1)
        _ = random_mps(shape, rank)

    # Not respecting the boundary rank conditions
    with np.testing.assert_raises(ValueError):
        shape = (10, 11, 12)
        rank = (1, 3, 3, 3)
        _ = random_mps(shape, rank)