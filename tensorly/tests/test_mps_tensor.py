import numpy as np

import tensorly as tl
import tensorly.backend as T
from tensorly.decomposition import matrix_product_state
from tensorly.testing import assert_array_almost_equal


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
