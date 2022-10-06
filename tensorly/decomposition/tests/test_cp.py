import numpy as np
import pytest

import tensorly as tl
from .._cp import (
    parafac,
    initialize_cp,
    sample_khatri_rao,
    randomised_parafac,
    CP,
    RandomizedCP,
)
from .._nn_cp import (
    non_negative_parafac,
    non_negative_parafac_hals,
    CP_NN,
    CP_NN_HALS,
)
from ...cp_tensor import cp_to_tensor
from ...cp_tensor import cp_to_tensor
from ...random import random_cp
from ...tenalg import khatri_rao
from ... import backend as T
from ...testing import (
    assert_array_equal,
    assert_,
    assert_class_wrapper_correctly_passes_arguments,
    assert_array_almost_equal,
)
from ...metrics.factors import congruence_coefficient


class ErrorTracker:
    def __init__(self):
        self.error = list()

    def __call__(self, cp_cur, rec_error):
        self.error.append(rec_error)


@pytest.mark.parametrize("linesearch", [True, False])
@pytest.mark.parametrize("orthogonalise", [True, False])
@pytest.mark.parametrize("true_rank,rank", [(1, 1), (3, 4)])
@pytest.mark.parametrize("init", ["svd", "random"])
@pytest.mark.parametrize("normalize_factors", [False, True])
@pytest.mark.parametrize("random_state", [1, 1234])
# mxnet does not support complex numbers. tensorflow has issues with type promotion that would require more code changes
@pytest.mark.parametrize(
    "complex", [False] if tl.get_backend() in ["mxnet", "tensorflow"] else [True, False]
)
def test_parafac(
    linesearch,
    orthogonalise,
    true_rank,
    rank,
    init,
    normalize_factors,
    random_state,
    complex,
    monkeypatch,
):
    """Test for the CANDECOMP-PARAFAC decomposition"""
    rng = tl.check_random_state(random_state)
    tol_norm_2 = 0.01
    tol_max_abs = 0.05
    shape = (6, 8, 7)

    factors = random_cp(
        shape,
        rank=true_rank,
        orthogonal=orthogonalise,
        full=False,
        random_state=rng,
        dtype=tl.float64,
    )

    # Generate a random complex tensor if requested
    if complex:
        factors_imag = random_cp(
            shape,
            rank=true_rank,
            orthogonal=orthogonalise,
            full=False,
            random_state=rng,
            dtype=tl.float64,
        )
        factors.factors = [
            fm_re + (fm_im * 1.0j)
            for fm_re, fm_im in zip(factors.factors, factors_imag.factors)
        ]

    tensor = tl.cp_to_tensor(factors)

    # Callback to record error
    errors = ErrorTracker()

    rng = tl.check_random_state(random_state)
    fac = parafac(
        tensor,
        rank=rank,
        n_iter_max=75,
        init=init,
        tol=1e-6,
        random_state=rng,
        normalize_factors=normalize_factors,
        orthogonalise=orthogonalise,
        linesearch=linesearch,
        callback=errors,
    )

    # Given all the random seed is set, this should provide the same answer for random initialization
    if init == "random":
        # Callback to record error
        errorsTwo = ErrorTracker()

        rng = tl.check_random_state(random_state)
        facTwo = parafac(
            tensor,
            rank=rank,
            n_iter_max=75,
            init=init,
            tol=1e-6,
            random_state=rng,
            normalize_factors=normalize_factors,
            orthogonalise=orthogonalise,
            linesearch=linesearch,
            callback=errorsTwo,
        )
        assert_array_almost_equal(errors.error, errorsTwo.error)
        assert_array_almost_equal(fac.factors[0], facTwo.factors[0])
        assert_array_almost_equal(fac.factors[1], facTwo.factors[1])
        assert_array_almost_equal(fac.factors[2], facTwo.factors[2])

    # Check that the error monotonically decreases
    if not orthogonalise:
        assert_(np.all(np.diff(errors.error) <= 1.0e-7))

    rec = cp_to_tensor(fac)
    error = T.norm(rec - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(
        error < tol_norm_2,
        f"norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}",
    )
    # Test the max abs difference between the reconstruction and the tensor
    assert_(
        T.max(T.abs(rec - tensor)) < tol_max_abs,
        f"abs norm of reconstruction error = {T.max(T.abs(rec - tensor))} higher than tolerance={tol_max_abs}",
    )

    # Test fixing mode 0 or 1 with given init
    fixed_tensor = random_cp(
        shape, rank=true_rank, normalise_factors=False, **tl.context(tensor)
    )
    rec_svd_fixed_mode_0 = parafac(
        tensor,
        rank=true_rank,
        n_iter_max=2,
        init=fixed_tensor,
        fixed_modes=[0],
        linesearch=linesearch,
    )
    rec_svd_fixed_mode_1 = parafac(
        tensor,
        rank=true_rank,
        n_iter_max=2,
        init=fixed_tensor,
        fixed_modes=[1],
        linesearch=linesearch,
    )
    # Check if modified after 2 iterations
    assert_array_equal(
        rec_svd_fixed_mode_0.factors[0],
        fixed_tensor.factors[0],
        err_msg="Fixed mode 0 was modified in candecomp_parafac",
    )
    assert_array_equal(
        rec_svd_fixed_mode_1.factors[1],
        fixed_tensor.factors[1],
        err_msg="Fixed mode 1 was modified in candecomp_parafac",
    )

    # Check that sparse component works
    rec_sparse, sparse_component = parafac(
        tensor,
        rank=rank,
        n_iter_max=200,
        init=init,
        tol=1.0e-6,
        sparsity=0.9,
        orthogonalise=orthogonalise,
        linesearch=linesearch,
    )

    rec_sparse = cp_to_tensor(rec_sparse) + sparse_component
    error = T.norm(rec_sparse - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2, "l2 Reconstruction error for sparsity!=None too high")
    assert_(
        T.max(T.abs(rec_sparse - tensor)) < tol_max_abs,
        "abs Reconstruction error for sparsity!=None too high",
    )

    with np.testing.assert_raises(ValueError):
        _, _ = initialize_cp(tensor, rank, init="bogus init type")

    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch, parafac, CP, ignore_args={"return_errors"}, rank=3
    )


@pytest.mark.parametrize("linesearch", [True, False])
def test_masked_parafac(linesearch):
    """Test for the masked CANDECOMP-PARAFAC decomposition.
    This checks that a mask of 1's is identical to the unmasked case.
    """
    tensor = random_cp((4, 4, 4), rank=1, full=True)
    mask = np.ones((4, 4, 4))
    mask[1, :, 3] = 0
    mask[:, 2, 3] = 0
    mask = tl.tensor(mask)
    tensor_mask = tensor * mask - 10000.0 * (1 - mask)

    fac = parafac(
        tensor_mask, svd_mask_repeats=0, mask=mask, n_iter_max=0, rank=1, init="svd"
    )
    fac_resvd = parafac(
        tensor_mask, svd_mask_repeats=10, mask=mask, n_iter_max=0, rank=1, init="svd"
    )
    err = tl.norm(tl.cp_to_tensor(fac) - tensor, 2)
    err_resvd = tl.norm(tl.cp_to_tensor(fac_resvd) - tensor, 2)
    assert_(err_resvd < err, "restarting SVD did not help")

    # Check that we get roughly the same answer with the full tensor and masking
    mask_fact = parafac(
        tensor,
        rank=1,
        mask=mask,
        init="random",
        random_state=1234,
        linesearch=linesearch,
    )
    fact = parafac(tensor, rank=1)
    diff = cp_to_tensor(mask_fact) - cp_to_tensor(fact)
    assert_(T.norm(diff) < 0.001, "norm 2 of reconstruction higher than 0.001")


def test_parafac_linesearch():
    """Test that we more rapidly converge to a solution with line search."""
    rng = tl.check_random_state(1234)
    eps = 10e-2
    tensor = T.tensor(rng.random_sample((5, 5, 5)))
    kt = parafac(
        tensor, rank=5, init="random", random_state=1234, n_iter_max=10, tol=10e-9
    )
    rec = tl.cp_to_tensor(kt)
    kt_ls = parafac(
        tensor,
        rank=5,
        init="random",
        random_state=1234,
        n_iter_max=10,
        tol=10e-9,
        linesearch=True,
    )
    rec_ls = tl.cp_to_tensor(kt_ls)

    rec_error = T.norm(tensor - rec) / T.norm(tensor)
    rec_error_ls = T.norm(tensor - rec_ls) / T.norm(tensor)
    assert_(
        rec_error_ls - rec_error < eps,
        f"Relative reconstruction error with line-search={rec_error_ls} VS {rec_error} without."
        "CP with line-search seems to have converged more slowly.",
    )


def test_non_negative_parafac(monkeypatch):
    """Test for non-negative PARAFAC

    TODO: more rigorous test
    """
    tol_norm_2 = 10e-1
    tol_max_abs = 1
    rng = tl.check_random_state(1234)
    tensor = T.tensor(rng.random_sample((3, 3, 3)) + 1)
    res = parafac(tensor, rank=3, n_iter_max=120)
    nn_res = non_negative_parafac(
        tensor, rank=3, n_iter_max=100, tol=10e-4, init="svd", verbose=0
    )

    # Make sure all components are positive
    _, nn_factors = nn_res
    for factor in nn_factors:
        assert_(T.all(factor >= 0))

    reconstructed_tensor = cp_to_tensor(res)
    nn_reconstructed_tensor = cp_to_tensor(nn_res)
    error = T.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= T.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2, "norm 2 of reconstruction higher than tol")

    # Test the max abs difference between the reconstruction and the tensor
    assert_(
        T.max(T.abs(reconstructed_tensor - nn_reconstructed_tensor)) < tol_max_abs,
        "abs norm of reconstruction error higher than tol",
    )

    # Test normalization
    nn_res = non_negative_parafac(
        tensor, rank=3, normalize_factors=True, tol=10e-4, init="svd"
    )
    nn_reconstructed_tensor = cp_to_tensor(nn_res)
    error = T.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= T.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2, "norm 2 of reconstruction higher than tol")

    # Test fixing mode 0 or 1 with given init
    fixed_tensor = random_cp((3, 3, 3), rank=2)
    for factor in fixed_tensor[1]:
        factor = T.abs(factor)
    rec_svd_fixed_mode_0 = non_negative_parafac(
        tensor, rank=2, n_iter_max=2, init=fixed_tensor, fixed_modes=[0]
    )
    rec_svd_fixed_mode_1 = non_negative_parafac(
        tensor, rank=2, n_iter_max=2, init=fixed_tensor, fixed_modes=[1]
    )
    # Check if modified after 2 iterations
    assert_array_equal(
        rec_svd_fixed_mode_0.factors[0],
        fixed_tensor.factors[0],
        err_msg="Fixed mode 0 was modified in candecomp_parafac",
    )
    assert_array_equal(
        rec_svd_fixed_mode_1.factors[1],
        fixed_tensor.factors[1],
        err_msg="Fixed mode 1 was modified in candecomp_parafac",
    )

    res_svd = non_negative_parafac(
        tensor, rank=3, n_iter_max=100, tol=10e-4, init="svd"
    )
    res_random = non_negative_parafac(
        tensor,
        rank=3,
        n_iter_max=100,
        tol=10e-4,
        init="random",
        random_state=1234,
        verbose=0,
    )
    rec_svd = cp_to_tensor(res_svd)
    rec_random = cp_to_tensor(res_random)
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    assert_(
        error < tol_norm_2, "norm 2 of difference between svd and random init too high"
    )
    assert_(
        T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
        "abs norm of difference between svd and random init too high",
    )

    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch, non_negative_parafac, CP_NN, ignore_args={"return_errors"}, rank=3
    )


def test_non_negative_parafac_hals(monkeypatch):
    """Test for non-negative PARAFAC HALS
    TODO: more rigorous test
    """
    tol_norm_2 = 10e-1
    tol_max_abs = 1
    rng = tl.check_random_state(1234)
    tensor = tl.tensor(rng.random_sample((3, 3, 3)) + 1)
    res = parafac(tensor, rank=3, n_iter_max=120)
    nn_res = non_negative_parafac_hals(
        tensor, rank=3, n_iter_max=100, tol=10e-4, init="svd", verbose=0
    )

    # Make sure all components are positive
    _, nn_factors = nn_res
    for factor in nn_factors:
        assert_(tl.all(factor >= 0))

    reconstructed_tensor = tl.cp_to_tensor(res)
    nn_reconstructed_tensor = tl.cp_to_tensor(nn_res)
    error = tl.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= tl.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2, "norm 2 of reconstruction higher than tol")

    # Test the max abs difference between the reconstruction and the tensor
    assert_(
        tl.max(tl.abs(reconstructed_tensor - nn_reconstructed_tensor)) < tol_max_abs,
        "abs norm of reconstruction error higher than tol",
    )

    # Test normalization
    nn_res = non_negative_parafac_hals(
        tensor, rank=3, normalize_factors=True, tol=10e-4, init="svd"
    )
    nn_reconstructed_tensor = cp_to_tensor(nn_res)
    error = T.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= T.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2, "norm 2 of reconstruction higher than tol")

    # Test fixing mode 0 or 1 with given init
    fixed_tensor = random_cp((3, 3, 3), rank=2)
    for factor in fixed_tensor[1]:
        factor = tl.abs(factor)
    rec_svd_fixed_mode_0 = non_negative_parafac_hals(
        tensor, rank=2, n_iter_max=2, init=fixed_tensor, fixed_modes=[0]
    )
    rec_svd_fixed_mode_1 = non_negative_parafac_hals(
        tensor, rank=2, n_iter_max=2, init=fixed_tensor, fixed_modes=[1]
    )
    # Check if modified after 2 iterations
    assert_array_equal(
        rec_svd_fixed_mode_0.factors[0],
        fixed_tensor.factors[0],
        err_msg="Fixed mode 0 was modified in candecomp_parafac",
    )
    assert_array_equal(
        rec_svd_fixed_mode_1.factors[1],
        fixed_tensor.factors[1],
        err_msg="Fixed mode 1 was modified in candecomp_parafac",
    )

    res_svd = non_negative_parafac_hals(
        tensor, rank=3, n_iter_max=100, tol=10e-4, init="svd"
    )
    res_random = non_negative_parafac_hals(
        tensor, rank=3, n_iter_max=100, tol=10e-4, init="random", verbose=0
    )
    rec_svd = tl.cp_to_tensor(res_svd)
    rec_random = tl.cp_to_tensor(res_random)
    error = tl.norm(rec_svd - rec_random, 2)
    error /= tl.norm(rec_svd, 2)
    assert_(
        error < tol_norm_2, "norm 2 of difference between svd and random init too high"
    )
    assert_(
        tl.max(tl.abs(rec_svd - rec_random)) < tol_max_abs,
        "abs norm of difference between svd and random init too high",
    )

    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch,
        non_negative_parafac_hals,
        CP_NN_HALS,
        ignore_args={"return_errors"},
        rank=3,
    )

    # Regression test: used wrong variable for convergence checking
    # Used mttkrp*factor instead of mttkrp*factors[-1], which resulted in
    # error when mode 2 was not constrained and erroneous convergence checking
    # when mode 2 was constrained.
    tensor = tl.tensor(rng.random_sample((3, 3, 3)) + 1)
    nn_estimate, errs = non_negative_parafac_hals(
        tensor,
        rank=2,
        n_iter_max=2,
        tol=1e-10,
        init="svd",
        verbose=0,
        nn_modes={
            0,
        },
        return_errors=True,
    )


def test_non_negative_parafac_hals_one_unconstrained():
    """Test for non-negative PARAFAC HALS
    TODO: more rigorous test
    """
    rng = tl.check_random_state(1234)
    t_shape = (8, 9, 10)
    rank = 3
    weights = T.tensor(rng.uniform(size=rank))
    A = T.tensor(rng.uniform(size=(t_shape[0], rank)))
    B = T.tensor(rng.standard_normal(size=(t_shape[1], rank)))
    C = T.tensor(rng.uniform(0.1, 1.1, size=(t_shape[2], rank)))
    cp_tensor = (weights, (A, B, C))
    X = cp_to_tensor(cp_tensor)

    nn_estimate, errs = non_negative_parafac_hals(
        X,
        rank=3,
        n_iter_max=100,
        tol=0,
        init="svd",
        verbose=0,
        nn_modes={0, 2},
        return_errors=True,
    )
    X_hat = cp_to_tensor(nn_estimate)
    assert_(
        tl.norm(
            X - X_hat,
        )
        < 1e-3,
        "Error was too high",
    )

    assert_(
        congruence_coefficient(A, nn_estimate[1][0], absolute_value=True)[0] > 0.99,
        "Factor recovery not high enough",
    )
    assert_(
        congruence_coefficient(B, nn_estimate[1][1], absolute_value=True)[0] > 0.99,
        "Factor recovery not high enough",
    )
    assert_(
        congruence_coefficient(C, nn_estimate[1][2], absolute_value=True)[0] > 0.99,
        "Factor recovery not high enough",
    )

    assert_(T.all(nn_estimate[1][0] > -1e-10))
    assert_(T.all(nn_estimate[1][2] > -1e-10))


@pytest.mark.xfail(tl.get_backend() == "tensorflow", reason="Fails on tensorflow")
def test_sample_khatri_rao():
    """Test for sample_khatri_rao"""
    rng = tl.check_random_state(1234)
    t_shape = (8, 9, 10)
    rank = 3
    _, factors = random_cp(t_shape, rank, full=False)
    num_samples = 4
    skip_matrix = 1
    sampled_kr, _, sampled_rows = sample_khatri_rao(
        factors,
        num_samples,
        skip_matrix=skip_matrix,
        return_sampled_rows=True,
        random_state=rng,
    )
    assert_(
        T.shape(sampled_kr) == (num_samples, rank),
        "Sampled shape of khatri-rao product is inconsistent",
    )
    assert_(
        np.max(sampled_rows) < (t_shape[0] * t_shape[2]),
        "Largest sampled row index is bigger than number of columns of"
        "unfolded matrix",
    )
    assert_(
        np.min(sampled_rows) >= 0, "Smallest sampled row index index is smaller than 0"
    )
    true_kr = khatri_rao(factors, skip_matrix=skip_matrix)
    for ix, j in enumerate(sampled_rows):
        assert_array_equal(
            true_kr[j],
            sampled_kr[int(ix)],
            err_msg="Sampled khatri_rao product doesnt correspond to product",
        )


@pytest.mark.xfail(tl.get_backend() == "tensorflow", reason="Fails on tensorflow")
def test_randomised_parafac(monkeypatch):
    """Test for randomised_parafac"""
    rng = tl.check_random_state(1234)
    t_shape = (10, 10, 10)
    n_samples = 8
    tensor = T.tensor(rng.random_sample(t_shape))
    rank = 4
    _, factors_svd = randomised_parafac(
        tensor, rank, n_samples, n_iter_max=1000, init="svd", tol=10e-5, verbose=True
    )
    for i, f in enumerate(factors_svd):
        assert_(T.shape(f) == (t_shape[i], rank), "Factors are of incorrect size")

    # test tensor reconstructed properly
    tolerance = 0.05
    tensor = random_cp(shape=(10, 10, 10), rank=4, full=True)
    cp_tensor = randomised_parafac(
        tensor,
        rank=5,
        n_samples=100,
        max_stagnation=20,
        n_iter_max=100,
        tol=0,
        verbose=0,
    )
    reconstruction = cp_to_tensor(cp_tensor)
    error = float(T.norm(reconstruction - tensor, 2) / T.norm(tensor, 2))
    assert_(
        error < tolerance,
        msg="reconstruction of {} (higher than tolerance of {})".format(
            error, tolerance
        ),
    )

    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch,
        randomised_parafac,
        RandomizedCP,
        ignore_args={"return_errors"},
        rank=3,
        n_samples=100,
    )
