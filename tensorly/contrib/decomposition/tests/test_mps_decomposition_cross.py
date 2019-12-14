import tensorly as tl

import pytest
import numpy as np
import itertools
import numpy.random as npr

from ..mps_decomposition_cross import matrix_product_state_cross
from ....mps_tensor import mps_to_tensor
from ....random import check_random_state
from tensorly.testing import assert_

skip_if_tensorflow = pytest.mark.skipif(tl.get_backend() == "tensorflow",
                                        reason="Operation not supported in TensorFlow")

@skip_if_tensorflow
def test_matrix_product_state_cross_1():
    """ Test for matrix_product_state """
    rng = check_random_state(1234)

    ## Test 1

    # Create tensor with random elements
    d = 3
    n = 4
    tensor = (np.arange(n**d).reshape((n,)*d))
    tensor = tl.tensor(tensor)


    tensor_shape = tensor.shape

    # Find MPS decomposition of the tensor
    rank = [1, 3,3, 1]
    factors = matrix_product_state_cross(tensor, rank, tol=1e-5, n_iter_max=10)
    assert(len(factors) == d), "Number of factors should be 4, currently has " + str(len(factors))

    # Check that the ranks are correct and that the second mode of each factor
    # has the correct number of elements
    r_prev_iteration = 1
    for k in range(d):
        (r_prev_k, n_k, r_k) = factors[k].shape
        assert(tensor_shape[k] == n_k), "Mode 1 of factor " + str(k) + "needs " + str(tensor_shape[k]) + " dimensions, currently has " + str(n_k)
        assert(r_prev_k == r_prev_iteration), " Incorrect ranks of factors "
        r_prev_iteration = r_k

@skip_if_tensorflow
def test_matrix_product_state_cross_2():
    """ Test for matrix_product_state """
    rng = check_random_state(1234)

    ## Test 2
    # Create tensor with random elements
    tensor = tl.tensor(rng.random_sample([3, 4, 5, 6, 2, 10]))
    tensor_shape = tensor.shape

    # Find MPS decomposition of the tensor
    rank = [1, 3, 3, 4, 2, 2, 1]
    factors = matrix_product_state_cross(tensor, rank)

    for k in range(6):
        (r_prev, n_k, r_k) = factors[k].shape

        first_error_message = "MPS rank " + str(k) + " is greater than the maximum allowed "
        first_error_message += str(r_prev) + " > " + str(rank[k])
        assert(r_prev<=rank[k]), first_error_message

        first_error_message = "MPS rank " + str(k+1) + " is greater than the maximum allowed "
        first_error_message += str(r_k) + " > " + str(rank[k+1])
        assert(r_k<=rank[k+1]), first_error_message

@skip_if_tensorflow
def test_matrix_product_state_cross_3():
    """ Test for matrix_product_state """
    rng = check_random_state(1234)

    ## Test 3
    tol = 10e-5
    tensor = tl.tensor(rng.random_sample([3, 3, 3]))
    factors = matrix_product_state_cross(tensor, (1, 3, 3, 1))
    reconstructed_tensor = mps_to_tensor(factors)
    error = tl.norm(reconstructed_tensor - tensor, 2)
    error /= tl.norm(tensor, 2)
    assert_(error < tol,
              'norm 2 of reconstruction higher than tol')

@skip_if_tensorflow
def test_matrix_product_state_cross_4():
    """ Test for matrix_product_state """

    # TEST 4
    # Random tensor is not really compress-able. Test on a tensor as values of a function

    def getEquispaceGrid(n_dim, rng, subdivisions):
        '''
        Returns a grid of equally-spaced points in the specified number of dimensions

        n_dim       : The number of dimensions to construct the tensor grid in
        rng         : The maximum dimension coordinate (grid starts at 0)
        subdivisions: Number of subdivisions of the grid to construct
        '''

        return np.array([np.array(range(subdivisions + 1)) * rng * 1.0 / subdivisions for i in range(n_dim)])

    def evaluateGrid(grid, fcn):
        '''
        Loops over a grid in specified order and computes the specified function at each
        point in the grid, returning a list of computed values.
        '''
        d, n = grid.shape
        values = np.zeros(len(grid[0]) ** len(grid))
        idx = 0
        for permutation in itertools.product(range(len(grid[0])), repeat=len(grid)):
            pt = np.array([grid[i][permutation[i]] for i in range(len(permutation))])
            values[idx] = fcn(pt)
            idx += 1

        return values.reshape((n,)*d)

    def func (X):
        return sum(X)**3

    maxvoleps = 1e-4
    tol = 1e-3
    n = 10
    d = 4
    rng = 1
    grid = getEquispaceGrid(d, rng, n)
    value = evaluateGrid(grid, func)
    value = tl.tensor(value)

    # Find MPS decomposition of the tensor
    rank = [1, 4, 4, 4, 1]
    factors = matrix_product_state_cross(value, rank, tol=tol)

    approx = mps_to_tensor(factors)
    error = tl.norm(approx-value,2)
    error /= tl.norm(value, 2)

    print(error)
    assert_(error < 1e-5, 'norm 2 of reconstruction higher than tol')
