import numpy as np

import tensorly as tl
from ..decomposition import matrix_product_state
from ..mps_tensor import mps_to_tensor, _validate_mps_tensor
from ..testing import assert_array_almost_equal, assert_equal, assert_raises
from ..random import check_random_state, random_mps


def test_validate_mps_tensor():
    rng = check_random_state(12345)
    true_shape = (3, 4, 5)
    true_rank = (1, 3, 2, 1)
    factors = random_mps(true_shape, rank=true_rank)
    
    # Check that the correct shape/rank are returned
    shape, rank = _validate_mps_tensor(factors)
    assert_equal(shape, true_shape,
                    err_msg='Returned incorrect shape (got {}, expected {})'.format(
                        shape, true_shape))
    assert_equal(rank, true_rank,
                    err_msg='Returned incorrect rank (got {}, expected {})'.format(
                        rank, true_rank))

    
    # One of the factors has the wrong ndim
    factors[0] = tl.tensor(rng.random_sample((4, 4)))
    with assert_raises(ValueError):
        _validate_mps_tensor(factors)
    
    # Consecutive factors ranks don't match
    factors[0] = tl.tensor(rng.random_sample((1, 3, 2)))
    with assert_raises(ValueError):
        _validate_mps_tensor(factors)
        
    # Boundary conditions not respected
    factors[0] = tl.tensor(rng.random_sample((3, 3, 2)))
    with assert_raises(ValueError):
        _validate_mps_tensor(factors)

    # Not enough factors
    with assert_raises(ValueError):
        _validate_mps_tensor(factors[:1])


def test_mps_to_tensor():
    """ Test for mps_to_tensor

        References
        ----------
        .. [1] Anton Rodomanov. "Introduction to the Tensor Train Decomposition
           and Its Applications in Machine Learning", HSE Seminar on Applied
           Linear Algebra, Moscow, Russia, March 2016.
    """

    # Create tensor
    n1 = 3
    n2 = 4
    n3 = 2

    tensor = np.zeros((n1, n2, n3))

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                tensor[i][j][k] = (i+1) + (j+1) + (k+1)

    tensor = tl.tensor(tensor)

    # Compute ground truth MPS factors
    factors = [None] * 3

    factors[0] = np.zeros((1, 3, 2))
    factors[1] = np.zeros((2, 4, 2))
    factors[2] = np.zeros((2, 2, 1))

    for i in range(3):
        for j in range(4):
            for k in range(2):
                factors[0][0][i][0] = i+1
                factors[0][0][i][1] = 1

                factors[1][0][j][0] = 1
                factors[1][0][j][1] = 0
                factors[1][1][j][0] = j+1
                factors[1][1][j][1] = 1

                factors[2][0][k][0] = 1
                factors[2][1][k][0] = k+1

    factors = [tl.tensor(f) for f in factors]

    # Check that MPS factors re-assemble to the original tensor
    assert_array_almost_equal(tensor, tl.mps_to_tensor(factors))


def test_mps_to_tensor_random():
    """ Test for mps_to_tensor

        Uses random tensor as input
    """

    # Create tensor with random elements
    tensor = tl.tensor(np.random.rand(3, 4, 5, 6, 2, 10))
    tensor_shape = tensor.shape

    # Find MPS decomposition of the tensor
    rank = 10
    factors = matrix_product_state(tensor, rank)

    # Reconstruct the original tensor
    reconstructed_tensor = tl.mps_to_tensor(factors)

    # Check that the rank is 10
    D = len(factors)
    for k in range(D):
        (r_prev, n_k, r_k) = factors[k].shape
        assert(r_prev<=rank), "MPS rank with index " + str(k) + "exceeds rank"
        assert(r_k<=rank), "MPS rank with index " + str(k+1) + "exceeds rank"
