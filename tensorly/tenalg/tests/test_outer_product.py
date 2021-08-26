import tensorly as tl
from tensorly import testing
from tensorly import random
from tensorly import tenalg

from .. import outer, batched_outer


def test_outer_product():
    """Test outer_dot"""
    rng = tl.check_random_state(1234)

    X = tl.tensor(rng.random_sample((4, 5, 6)))
    Y = tl.tensor(rng.random_sample((3, 4)))
    Z = tl.tensor(rng.random_sample((2)))
    tdot = outer([X, Y, Z])
    true_dot = tenalg.tensordot(X, Y, ())
    true_dot = tenalg.tensordot(true_dot, Z, ())
    testing.assert_array_almost_equal(tdot, true_dot)


def test_batched_outer_product():
    """Test batched_outer_dot

    Notes
    -----
    At the time of writing, MXNet doesn't support transpose 
    for tensors of order higher than 6
    """
    rng = tl.check_random_state(1234)
    batch_size = 3
    
    X = tl.tensor(rng.random_sample((batch_size, 4, 5, 6)))
    Y = tl.tensor(rng.random_sample((batch_size, 3)))
    Z = tl.tensor(rng.random_sample((batch_size, 2)))
    res = batched_outer([X, Y, Z])
    true_res = tenalg.tensordot(X, Y, (), batched_modes=0)
    true_res = tenalg.tensordot(true_res, Z, (), batched_modes=0)

    testing.assert_array_almost_equal(res, true_res)
