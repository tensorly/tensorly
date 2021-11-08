import pytest
import tensorly as tl
from .._tt import tensor_train, tensor_train_matrix, TensorTrain
from ...tt_matrix import tt_matrix_to_tensor
from ...random import random_tt
from ...testing import assert_, assert_array_almost_equal, assert_class_wrapper_correctly_passes_arguments

skip_mxnet= pytest.mark.skipif(tl.get_backend() == "mxnet", 
                 reason="MXNet currently does not support transpose for tensors of order > 6.")

def test_tensor_train(monkeypatch):
    """ Test for tensor_train """
    rng = tl.check_random_state(1234)

    ## Test 1

    # Create tensor with random elements
    tensor = tl.tensor(rng.random_sample([3, 4, 5, 6, 2, 10]))
    tensor_shape = tensor.shape

    # Find TT decomposition of the tensor
    rank = [1, 3, 3, 4, 2, 2, 1]
    factors = tensor_train(tensor, rank)

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

    # Find TT decomposition of the tensor
    rank = [1, 5, 4, 3, 8, 10, 1]
    factors = tensor_train(tensor, rank)

    for k in range(6):
        (r_prev, n_k, r_k) = factors[k].shape

        first_error_message = "TT rank " + str(k) + " is greater than the maximum allowed "
        first_error_message += str(r_prev) + " > " + str(rank[k])
        assert(r_prev<=rank[k]), first_error_message

        first_error_message = "TT rank " + str(k+1) + " is greater than the maximum allowed "
        first_error_message += str(r_k) + " > " + str(rank[k+1])
        assert(r_k<=rank[k+1]), first_error_message

    ## Test 3
    tol = 10e-5
    tensor = tl.tensor(rng.random_sample([3, 3, 3]))
    factors = tensor_train(tensor, (1, 3, 3, 1))
    reconstructed_tensor = tl.tt_to_tensor(factors)
    error = tl.norm(reconstructed_tensor - tensor, 2)
    error /= tl.norm(tensor, 2)
    assert_(error < tol,
              'norm 2 of reconstruction higher than tol')

    assert_class_wrapper_correctly_passes_arguments(monkeypatch, tensor_train, TensorTrain, ignore_args={}, rank=3)


 # TODO: Remove once MXNet supports transpose for > 6th order tensors
@skip_mxnet

def test_tensor_train_matrix():
    """Test for tensor_train_matrix decomposition"""
    tensor = random_tt((2, 2, 2, 3, 3, 3), rank=2, full=True)
    tt = tensor_train_matrix(tensor, 10)

    tt_rec = tt_matrix_to_tensor(tt)
    assert_array_almost_equal(tensor, tt_rec, decimal=4)
