import pytest
import tensorly as tl
import numpy as np

from .._tr import (
    tensor_ring,
    tensor_ring_als,
    tensor_ring_als_sampled,
    TensorRingALS,
    TensorRingALSSampled,
)
from ...random import random_tr
from ...testing import (
    assert_,
    assert_array_almost_equal,
    assert_raises,
    assert_class_wrapper_correctly_passes_arguments,
)


class ErrorTracker:
    def __init__(self):
        self.error = list()

    def __call__(self, decomp, rec_error):
        self.error.append(tl.to_numpy(rec_error))


def test_tensor_ring():
    """Test for tensor_ring"""
    # Create tensor with random elements
    tensor_shape = (6, 2, 3, 2, 6)
    rank = (3, 2, 4, 12, 18, 3)
    tensor = random_tr(tensor_shape, rank, full=True, random_state=1234)

    # Compute TR decomposition
    tr_tensor = tensor_ring(tensor, rank)
    assert_(
        len(tr_tensor.factors) == len(tensor_shape),
        f"Number of factors should be {len(tensor_shape)}, currently has {len(tr_tensor.factors)}",
    )

    for k in range(len(tensor_shape)):
        (r_prev_k, n_k, r_k) = tr_tensor[k].shape
        assert_(
            n_k == tensor_shape[k],
            f"Mode 2 of factor {k} should have {tensor_shape[k]} dimensions, currently has {n_k}",
        )
        assert_(r_prev_k == rank[k], "Incorrect ranks")
        if k:
            assert_(r_prev_k == r_prev_iteration, "Incorrect ranks")
        r_prev_iteration = r_k

    assert_array_almost_equal(tr_tensor.to_tensor(), tensor, decimal=2)


def test_tensor_ring_mode():
    """Test for tensor_ring `mode` argument"""
    # Create tensor with random elements
    tensor_shape = (6, 2, 3, 2, 6)
    rank = (12, 2, 1, 3, 6, 12)
    tensor = random_tr(tensor_shape, rank, full=True, random_state=1234)

    # Compute TR decomposition
    tr_tensor = tensor_ring(tensor, rank, mode=1)
    assert_(
        len(tr_tensor.factors) == len(tensor_shape),
        f"Number of factors should be {len(tensor_shape)}, currently has {len(tr_tensor.factors)}",
    )

    for k in range(len(tensor_shape)):
        (r_prev_k, n_k, r_k) = tr_tensor[k].shape
        assert_(
            n_k == tensor_shape[k],
            f"Mode 2 of factor {k} should have {tensor_shape[k]} dimensions, currently has {n_k}",
        )
        assert_(r_prev_k == rank[k], "Incorrect ranks")
        if k:
            assert_(r_prev_k == r_prev_iteration, "Incorrect ranks")
        r_prev_iteration = r_k

    assert_array_almost_equal(tr_tensor.to_tensor(), tensor, decimal=2)

    with assert_raises(ValueError):
        tensor_ring(tensor, rank=(12, 2, 10, 3, 6, 12), mode=1)


@pytest.mark.parametrize(
    "tensor_shape, rank",
    [((6, 2, 3, 2, 6), (3, 2, 4, 12, 18, 3)), ((20, 18, 19), (6, 7, 8, 6))],
)
@pytest.mark.parametrize(
    "ls_solve, er_decrease_tol", [("lstsq", 1e-7), ("normal_eq", 1e-3)]
)  # Lower tolerance for normal equation approach due to lower accuracy in solves
@pytest.mark.parametrize("random_state", [1, 1234])
def test_tensor_ring_als(
    tensor_shape, rank, ls_solve, random_state, er_decrease_tol, monkeypatch
):
    rng = tl.check_random_state(random_state)

    # Generate random tensor which has exact tensor ring decomposition
    tensor = random_tr(
        tensor_shape, rank, full=True, random_state=rng, dtype=tl.float64
    )

    # Ensure ValueError is raised for when invalid ls_solve is given
    with np.testing.assert_raises(ValueError):
        tr_decomp = tensor_ring_als(
            tensor, rank, random_state=rng, ls_solve="invalid_ls_solve"
        )

    # Create callback function for error tracking and run decomposition
    callback = ErrorTracker()
    tr_decomp = tensor_ring_als(
        tensor, rank, random_state=rng, callback=callback, ls_solve=ls_solve
    )

    # Ensure decomposition returns right number of factors
    assert_(len(tr_decomp.factors) == len(tensor_shape))

    # Ensure cores are sized correctly
    for i in range(len(tensor_shape)):
        core_shape = tr_decomp[i].shape
        assert_(core_shape[0] == rank[i])
        assert_(core_shape[1] == tensor_shape[i])
        assert_(core_shape[2] == rank[i + 1])

    # Compute decomposition relative error and ensure it's small enough
    rel_error_tol = 1e-2
    rel_error = tl.norm(tl.tr_to_tensor(tr_decomp) - tensor) / tl.norm(tensor)
    assert_(rel_error < rel_error_tol)

    # Ensure error decreases monotonically (up to numerical error)
    for i in range(len(callback.error) - 1):
        assert_(callback.error[i + 1] < callback.error[i] + er_decrease_tol)

    # Ensure TensorRingALS class passes arguments correctly to decomposition function
    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch, tensor_ring_als, TensorRingALS, rank=3
    )

    # Ensure that the computed decomposition is the same when the same random seed is
    # used
    rng1 = tl.check_random_state(random_state)
    rng2 = tl.check_random_state(random_state)
    callback1 = ErrorTracker()
    callback2 = ErrorTracker()
    tr_decomp1 = tensor_ring_als(
        tensor, rank, random_state=rng1, callback=callback1, ls_solve=ls_solve
    )
    tr_decomp2 = tensor_ring_als(
        tensor, rank, random_state=rng2, callback=callback2, ls_solve=ls_solve
    )
    assert_array_almost_equal(callback1.error, callback2.error)
    for i in range(len(tr_decomp1.factors)):
        assert_array_almost_equal(tr_decomp1[i], tr_decomp2[i])


@pytest.mark.parametrize(
    "tensor_shape, rank, n_samples",
    [((6, 2, 3, 2, 6), (3, 2, 4, 12, 18, 3), 300), ((20, 18, 19), (6, 7, 8, 6), 300)],
)
@pytest.mark.parametrize("random_state", [1, 1234])
@pytest.mark.xfail(tl.get_backend() == "tensorflow", reason="Fails on tensorflow")
def test_tensor_ring_als_sampled(
    tensor_shape, rank, n_samples, random_state, monkeypatch
):
    rng = tl.check_random_state(random_state)

    # Generate random tensor which has exact tensor ring decomposition
    tensor = random_tr(
        tensor_shape, rank, full=True, random_state=rng, dtype=tl.float64
    )

    # Create callback function for error tracking and run decomposition
    callback = ErrorTracker()
    tr_decomp = tensor_ring_als_sampled(
        tensor=tensor,
        rank=rank,
        n_samples=n_samples,
        n_iter_max=100,
        tol=0,
        random_state=rng,
        callback=callback,
    )

    # Ensure decomposition returns right number of factors
    assert_(len(tr_decomp.factors) == len(tensor_shape))

    # Ensure cores are sized correctly
    for i in range(len(tensor_shape)):
        core_shape = tr_decomp[i].shape
        assert_(core_shape[0] == rank[i])
        assert_(core_shape[1] == tensor_shape[i])
        assert_(core_shape[2] == rank[i + 1])

    # Compute decomposition relative error and ensure it's small enough.
    # Note that sampling-based decomposition of the small tensors used in this test
    # is somewhat precarious, so the rel_error_tol or number of samples used may have
    # to be adapted in case of failure for other backends/random seeds.
    rel_error_tol = 1e-2
    rel_error = tl.norm(tl.tr_to_tensor(tr_decomp) - tensor) / tl.norm(tensor)
    assert_(
        rel_error < rel_error_tol,
        msg="Consider increasing number of samples used in test or changing seed.",
    )

    # Ensure TensorRingALS class passes arguments correctly to decomposition function
    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch,
        tensor_ring_als_sampled,
        TensorRingALSSampled,
        rank=3,
        n_samples=10,
    )

    # Ensure that the computed decomposition is the same when the same random seed is
    # used
    rng1 = tl.check_random_state(random_state)
    rng2 = tl.check_random_state(random_state)
    callback1 = ErrorTracker()
    callback2 = ErrorTracker()
    tr_decomp1 = tensor_ring_als_sampled(
        tensor=tensor,
        rank=rank,
        n_samples=n_samples,
        random_state=rng1,
        callback=callback1,
    )
    tr_decomp2 = tensor_ring_als_sampled(
        tensor=tensor,
        rank=rank,
        n_samples=n_samples,
        random_state=rng2,
        callback=callback2,
    )
    assert_array_almost_equal(callback1.error, callback2.error)
    for i in range(len(tr_decomp1.factors)):
        assert_array_almost_equal(tr_decomp1[i], tr_decomp2[i])


@pytest.mark.parametrize("randomized_error", [False, True])
@pytest.mark.xfail(tl.get_backend() == "tensorflow", reason="Fails on tensorflow")
def test_tensor_ring_als_sampled_large_decomp(randomized_error):
    # The point of this test is to attempt decomposing a sligthly larger tensor than the
    # test in test_tensor_ring_als_sampled. The tensor in the present function is
    # approaching a size where we can expect sampling to yield an accurate result even
    # with meaningful downsampling.

    rng = tl.check_random_state(1234)

    # Define tensor properties
    tensor_shape = [10, 11, 12, 13, 14]
    rank = [2, 3, 4, 5, 3, 2]

    # Some decomposition properties
    n_samples = 2000  # Note: The smallest least squares problem has 17160 rows
    n_iter_max = 100

    # Generate random tensor which has exact tensor ring decomposition
    tensor = random_tr(
        tensor_shape, rank, full=True, random_state=rng, dtype=tl.float64
    )

    # Create callback function for error tracking and run decomposition
    callback = ErrorTracker()
    tr_decomp = tensor_ring_als_sampled(
        tensor=tensor,
        rank=rank,
        n_samples=n_samples,
        n_iter_max=n_iter_max,
        tol=0,
        randomized_error=randomized_error,
        random_state=rng,
        callback=callback,
    )

    # Get relative error from callback object
    rel_error = callback.error[-1]

    # Check if computed relative error is less than acceptable tolerance
    rel_error_tol = 1e-7
    assert_(
        rel_error < rel_error_tol,
        msg="Consider increasing number of samples used in test or changing seed.",
    )
