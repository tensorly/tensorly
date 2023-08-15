import numpy as np

import tensorly as tl
from ..random import random_tr
from ..testing import assert_allclose, assert_equal, assert_raises, assert_
from ..tr_tensor import tr_to_tensor, _validate_tr_tensor, _tr_n_param, validate_tr_rank


def test_validate_tr_tensor():
    rng = tl.check_random_state(12345)
    true_shape = (6, 4, 5)
    true_rank = (3, 2, 2, 3)
    factors = random_tr(true_shape, rank=true_rank).factors

    # Check that the correct shape/rank are returned
    shape, rank = _validate_tr_tensor(factors)
    assert_equal(
        shape,
        true_shape,
        err_msg=f"Returned incorrect shape (got {shape}, expected {true_shape})",
    )
    assert_equal(
        rank,
        true_rank,
        err_msg=f"Returned incorrect rank (got {rank}, expected {true_rank})",
    )

    # One of the factors has the wrong ndim
    factors[0] = tl.tensor(rng.random_sample((4, 4)))
    with assert_raises(ValueError):
        _validate_tr_tensor(factors)

    # Consecutive factors ranks don't match
    factors[0] = tl.tensor(rng.random_sample((3, 6, 4)))
    with assert_raises(ValueError):
        _validate_tr_tensor(factors)

    # Boundary conditions not respected
    factors[0] = tl.tensor(rng.random_sample((2, 6, 2)))
    with assert_raises(ValueError):
        _validate_tr_tensor(factors)


def test_tr_to_tensor():
    # Create ground truth TR factors
    factors = [tl.randn((2, 4, 3)), tl.randn((3, 5, 2)), tl.randn((2, 6, 2))]

    # Create tensor
    tensor = tl.einsum("iaj,jbk,kci->abc", *factors)

    # Check that TR factors re-assemble to the original tensor
    assert_allclose(tensor, tr_to_tensor(factors), atol=1e-6, rtol=1e-6)


def test_validate_tr_rank():
    """Test for validate_tr_rank with random sizes"""
    tensor_shape = tuple(np.random.randint(1, 100, size=4))
    n_param_tensor = np.prod(tensor_shape)

    # Rounding = floor
    rank = validate_tr_rank(tensor_shape, rank="same", rounding="floor")
    n_param = _tr_n_param(tensor_shape, rank)
    assert_(n_param <= n_param_tensor)

    # Rounding = ceil
    rank = validate_tr_rank(tensor_shape, rank="same", rounding="ceil")
    n_param = _tr_n_param(tensor_shape, rank)
    assert_(n_param >= n_param_tensor)

    # Integer rank
    with assert_raises(ValueError):
        validate_tr_rank(tensor_shape, rank=(2, 3, 4, 2))

    with assert_raises(ValueError):
        validate_tr_rank(tensor_shape, rank=(2, 3, 4, 2, 3))
