import tensorly as tl

import pytest
import numpy as np
import itertools

from .._gcp import (gcp, tt_gcp_fg, vec2factors, factors2vec, validate_opt, validate_type)
from ....cp_tensor import (cp_to_tensor, CPTensor, cp_norm)

from tensorly.testing import assert_

skip_if_backend = pytest.mark.skipif(tl.get_backend() in ("tensorflow", "jax", "cupy", "mxnet"),
                                     reason=f"Operation not supported in {tl.get_backend()}")


@skip_if_backend
def test_gcp_1():
    """ Test for generalized CP"""

    ## Test 1 - shapes and dimensions

    # Create tensor with random elements
    rng = tl.check_random_state(1234)
    d = 3
    n = 4
    tensor = tl.tensor(rng.random((4, 5, 6)), dtype=tl.float32)
    # tensor = (np.arange(n**d, dtype=float).reshape((n,)*d))
    # tensor = tl.tensor(tensor)  # a 4 x 4 x 4 tensor

    tensor_shape = tensor.shape


    # Find gcp decomposition of the tensor
    rank = 2
    mTen = gcp(tensor, rank, type='normal', state=rng)
    print(mTen)
    assert(len(mTen[1]) == d), "Number of factors should be 3, currently has " + str(len(mTen[1]))

    # Check each factor matrices has the correct number of columns
    for k in range(d):
        rows, columns = tl.shape(mTen[1][k])
        assert(columns == rank), "Factor matrix {} needs {} columns, but only has {}".format(i+1, rank, columns)

    # Check CPTensor has same number of elements as tensor
    assert(tensor.size == tl.cp_to_tensor(mTen).size), "Unequal number of tensor elements. Tensor: {} CPTensor: {}".format(tensor.size,tl.cp_to_tensor(mTen).size)


@skip_if_backend
# def test_validate_type_1():
#     """ Test for gcp helper method for setting up loss and gradient function handles """
#     ## Test 'normal' fh and gh returned
#     type = 'normal'
#     fh, gh, lb = validate_type(type)
#     assert(fh(4,2) == 4), "normal loss function for x = 4 & m =2 should equal 4 not {}".format(fh(4,2))


@skip_if_backend
def test_validate_opt_1():
    """ Tests for gcp helper method for verifying optimization method"""
    opt = 'lbfgsb'
    assert(validate_opt(opt) == 0), "L-BFGS-B optimization is implemented, this should return 0"
    opt = 'sgd'
    assert(validate_opt(opt) == 1), "SGD optimization is implemented, this should return 0"
    opt = 'blind-luck'
    assert (validate_opt(opt) == 1), "blind luck isnt an optimization method, this should return 0"


@skip_if_backend
def test_factors2vec_1():
    """ Test wrapper function from GCP paper"""
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)
    factors = []
    for i in range(3):
        f = tl.transpose(X[:][:][i])
        factors.append(f)
    vec = factors2vec(factors)
    for i in range(vec.size):
        assert(vec[i] == i), "Value at vec[i] = {} doesn't equal the index value i = {}".format(vec[i], i)

@skip_if_backend
def test_vec2factors_1():
    """ Test wrapper function from GCP paper"""
    rank = 2
    X = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=tl.float32)
    X_shp = tl.shape(X)
    X_cntx = tl.context(X)

    factors = []
    for i in range(3):
        f = tl.transpose(X[:][:][i])
        factors.append(f)
    vec1 = factors2vec(factors)

    M = vec2factors(vec1,X_shp, rank, X_cntx)
    vec2 = factors2vec(M[1])
    for i in range(X_shp[0]):
        assert(vec1[i] == vec2[i])

