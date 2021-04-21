import numpy as np
import pytest

from .. import backend as tl
from ..base import unfold, tensor_to_vec
from ..parafac2_tensor import (parafac2_to_tensor, parafac2_to_unfolded,
                               parafac2_to_vec, _validate_parafac2_tensor,
                               parafac2_to_slice, parafac2_to_slices,
                               parafac2_normalise, apply_parafac2_projections)
from ..tenalg import kronecker, mode_dot
from ..testing import (assert_array_equal, assert_array_almost_equal, 
                       assert_equal, assert_raises)
from ..random import random_parafac2


def test_validate_parafac2_tensor():
    rng = tl.check_random_state(12345)
    true_shape = [(4, 5)]*3
    true_rank = 2
    weights, factors, projections = random_parafac2(true_shape, rank=true_rank)
    
    # Check shape and rank returned
    shape, rank = _validate_parafac2_tensor((weights, factors, projections))
    assert_equal(shape, true_shape,
                 err_msg='Returned incorrect shape (got {}, expected {})'.format(
                    shape, true_shape))
    assert_equal(rank, true_rank,
                 err_msg='Returned incorrect rank (got {}, expected {})'.format(
                    rank, true_rank))

    # One of the factors has the wrong rank
    for mode in range(3):
        false_shape = (tl.shape(factors[mode])[0], true_rank+1)
        factors[mode], copy = tl.tensor(rng.random_sample(false_shape)), factors[mode]
        with assert_raises(ValueError):
            _validate_parafac2_tensor((weights, factors, projections))
        
        factors[mode] = copy

    # Not three factor matrices
    with assert_raises(ValueError):
        _validate_parafac2_tensor((weights, factors[1:], projections))

    # Not enough projections
    with assert_raises(ValueError):
        _validate_parafac2_tensor((weights, factors, projections[1:]))
    
    # Wrong number of weights
    with assert_raises(ValueError):
        _validate_parafac2_tensor((weights[1:], factors, projections))
    
    # The projections aren't orthogonal
    false_projections = [tl.tensor(rng.random_sample(tl.shape(P))) for P in projections]
    with assert_raises(ValueError):
        _validate_parafac2_tensor((weights, factors, false_projections))


# Disable tests for inplace edits, since that possibility is removed
# to support TensorFlow.
#@pytest.mark.parametrize('copy', [True, False])
def test_parafac2_normalise():
    rng = tl.check_random_state(12345)
    true_shape = [(4, 5)]*3
    true_rank = 2
    parafac2_tensor = random_parafac2(true_shape, rank=true_rank)
    
    
    normalised_parafac2_tensor = parafac2_normalise(parafac2_tensor)  # , copy=copy)
    expected_norm = tl.ones(true_rank)
    for f in normalised_parafac2_tensor[1]:
        assert_array_almost_equal(tl.norm(f, axis=0), expected_norm)
    assert_array_almost_equal(parafac2_to_tensor(parafac2_tensor),
                              parafac2_to_tensor(normalised_parafac2_tensor))


def test_parafac2_to_tensor():
    weights = tl.tensor([2, 3])
    factors = [
        tl.tensor([[1, 1],
                   [1, 0]]),
        tl.tensor([[2, 1],
                   [1, 2]]),
        tl.tensor([[1, 1],
                   [1, 0],
                   [1, 0]])
    ]
    projections = [
        tl.tensor([[0, 0],
                   [1, 0],
                   [0, 1]]),
        tl.tensor([[1,  0],
                   [0,  0],
                   [0, -1]])
    ]

    true_res = tl.tensor(
        [[[ 0,  0,  0],
          [ 7,  4,  4],
          [ 8,  2,  2]],

         [[ 4,  4,  4],
          [ 0,  0,  0],
          [-2, -2, -2]]]
    )

    res = parafac2_to_tensor((weights, factors, projections))
    assert_array_equal(true_res, res)


def test_parafac2_to_slices():
    weights = tl.tensor([2, 3])
    factors = [
        tl.tensor([[1, 1],
                   [1, 0]]),
        tl.tensor([[2, 1],
                   [1, 2]]),
        tl.tensor([[1, 1],
                   [1, 0],
                   [1, 0]])
    ]
    projections = [
        tl.tensor([[1, 0],
                   [0, 1]]),
        tl.tensor([[1,  0],
                   [0,  0],
                   [0, -1]])
    ]
    true_res = [
        tl.tensor([[ 7,  4,  4],
                   [ 8,  2,  2]]),
        tl.tensor([[ 4,  4,  4],
                   [ 0,  0,  0],
                   [-2, -2, -2]])
    ]
    for i, true_slice in enumerate(true_res):
        assert_array_equal(parafac2_to_slice((weights, factors, projections), i), true_slice)
    
    for true_slice, est_slice in zip(true_res, parafac2_to_slices((weights, factors, projections))):
        assert_array_equal(true_slice, est_slice)



def test_parafac2_to_unfolded():
    """Test for parafac2_to_unfolded

    Notes
    -----
    Assumes that parafac2_to_tensor is properly tested
    """
    rng = tl.check_random_state(12345)
    true_shape = [(4, 5)]*3
    true_rank = 2
    pf2_tensor = random_parafac2(true_shape, true_rank)
    full_tensor = parafac2_to_tensor(pf2_tensor)
    for mode in range(tl.ndim(full_tensor)):
        assert_array_almost_equal(parafac2_to_unfolded(pf2_tensor, mode), unfold(full_tensor, mode))


def test_parafac2_to_vec():
    """Test for parafac2_to_vec

    Notes
    -----
    Assumes that parafac2_to_tensor works correctly
    """
    rng = tl.check_random_state(12345)
    true_shape = [(4, 5)]*3
    true_rank = 2
    pf2_tensor = random_parafac2(true_shape, true_rank)
    full_tensor = parafac2_to_tensor(pf2_tensor)
    assert_array_almost_equal(parafac2_to_vec(pf2_tensor), tensor_to_vec(full_tensor))


def test_apply_parafac2_projections():
    weights = tl.tensor([2, 3])
    factors = [
        tl.tensor([[1, 1],
                   [1, 0]]),
        tl.tensor([[2, 1],
                   [1, 2]]),
        tl.tensor([[1, 1],
                   [1, 0],
                   [1, 0]])
    ]
    projections = [
        tl.tensor([[1, 0],
                   [0, 1]]),
        tl.tensor([[1,  0],
                   [0,  0],
                   [0, -1]])
    ]
    true_res = [
        tl.tensor([[ 7,  4,  4],
                   [ 8,  2,  2]]),
        tl.tensor([[ 4,  4,  4],
                   [ 0,  0,  0],
                   [-2, -2, -2]])
    ]

    new_weights, projected_factors = apply_parafac2_projections((weights, factors, projections))

    assert_array_equal(new_weights, weights)
    for i, Bi in enumerate(projected_factors[1]):
        assert_array_almost_equal(tl.dot(projections[i], factors[1]), Bi)
