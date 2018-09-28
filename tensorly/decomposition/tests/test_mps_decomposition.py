import tensorly as tl
from tensorly.decomposition import matrix_product_state
from tensorly.random import check_random_state
from tensorly.testing import assert_


def test_matrix_product_state():
    """ Test for matrix_product_state """
    rng = check_random_state(1234)

    ## Test 1

    # Create tensor with random elements
    tensor = tl.tensor(rng.random_sample([3, 4, 5, 6, 2, 10]))
    tensor_shape = tensor.shape

    # Find MPS decomposition of the tensor
    rank = [1, 3, 3, 4, 2, 2, 1]
    factors = matrix_product_state(tensor, rank)

    assert(len(factors) == 6), "Number of factors should be 6, currently has " + str(len(factors))

    # Check that the ranks are correct and that the second mode of each factor
    # has the correct number of elements
    r_prev_iteration = 1
    for k in range(6):
        (r_prev_k, n_k, r_k) = factors[k].shape
        assert(tensor_shape[k] == n_k), "Mode 1 of factor " + str(k) + "needs " + str(tensor_shape[k]) + " dimensions, currently has " + str(n_k)
        assert(r_prev_k == r_prev_iteration), " Incorrect ranks of factors "
        r_prev_iteration = r_k

    ## Test 2
    # Create tensor with random elements
    tensor = tl.tensor(rng.random_sample([3, 4, 5, 6, 2, 10]))
    tensor_shape = tensor.shape

    # Find MPS decomposition of the tensor
    rank = [10, 5, 4, 3, 8, 10, 11]
    factors = matrix_product_state(tensor, rank)

    for k in range(6):
        (r_prev, n_k, r_k) = factors[k].shape

        first_error_message = "MPS rank " + str(k) + " is greater than the maximum allowed "
        first_error_message += str(r_prev) + " > " + str(rank[k])
        assert(r_prev<=rank[k]), first_error_message

        first_error_message = "MPS rank " + str(k+1) + " is greater than the maximum allowed "
        first_error_message += str(r_k) + " > " + str(rank[k+1])
        assert(r_k<=rank[k+1]), first_error_message

    ## Test 3
    tol = 10e-5
    tensor = tl.tensor(rng.random_sample([3, 3, 3]))
    factors = matrix_product_state(tensor, (1, 3, 3, 1))
    reconstructed_tensor = tl.mps_to_tensor(factors)
    error = tl.norm(reconstructed_tensor - tensor, 2)
    error /= tl.norm(tensor, 2)
    assert_(error < tol,
              'norm 2 of reconstruction higher than tol')
