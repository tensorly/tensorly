import tensorly as tl
from tensorly import testing
from tensorly import random
from tensorly import tenalg

from .. import tensor_dot, batched_tensor_dot


def test_tensor_product():
    """Test tensor_dot"""
    rng = tl.check_random_state(1234)

    X = tl.tensor(rng.random_sample((4, 5, 6)))
    Y = tl.tensor(rng.random_sample((3, 4, 7)))
    tdot = tl.tensor_to_vec(tensor_dot(X, Y))
    true_dot = tl.tensor_to_vec(tenalg.outer([tl.tensor_to_vec(X), tl.tensor_to_vec(Y)]))
    testing.assert_array_almost_equal(tdot, true_dot)   
    
def test_batched_tensor_product():
    """Test batched-tensor_dot

    Notes
    -----
    At the time of writing, MXNet doesn't support transpose 
    for tensors of order higher than 6
    """
    rng = tl.check_random_state(1234)
    batch_size = 3
    
    X = tl.tensor(rng.random_sample((batch_size, 4, 5, 6)))
    Y = tl.tensor(rng.random_sample((batch_size, 3, 7)))
    tdot = tl.unfold(batched_tensor_dot(X, Y), 0)
    for i in range(batch_size):
        true_dot = tl.tensor_to_vec(tenalg.outer([tl.tensor_to_vec(X[i]), tl.tensor_to_vec(Y[i])]))
        testing.assert_array_almost_equal(tdot[i], true_dot)
