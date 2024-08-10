import pytest

import tensorly as tl
from ...random import random_parafac2
from ...testing import (
    assert_,
    assert_class_wrapper_correctly_passes_arguments,
    assert_array_almost_equal,
    assert_allclose,
)
from .._parafac2 import (
    Parafac2,
    parafac2,
    initialize_decomposition,
    _BroThesisLineSearch,
)
from ...parafac2_tensor import Parafac2Tensor, parafac2_to_tensor, parafac2_to_slices
from ...metrics.factors import congruence_coefficient


@pytest.mark.parametrize("normalize_factors", [True, False])
@pytest.mark.parametrize("init", ["random", "svd"])
@pytest.mark.parametrize("linesearch", [False, True])
def test_parafac2(monkeypatch, normalize_factors, init, linesearch):
    rng = tl.check_random_state(1234)
    tol_norm_2 = 10e-2
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(25)],
        rank=rank,
        random_state=rng,
    )
    # It is difficult to correctly identify B[i, :, r] if A[i, r] is small.
    # This is sensible, since then B[i, :, r] contributes little to the total value of X.
    # To test the PARAFAC2 decomposition in the precence of roundoff errors, we therefore add
    # 0.01 to the A factor matrix.
    random_parafac2_tensor.factors[0] = random_parafac2_tensor.factors[0] + 0.01

    tensor = parafac2_to_tensor(random_parafac2_tensor)
    slices = parafac2_to_slices(random_parafac2_tensor)

    rec, err = parafac2(
        slices,
        rank,
        random_state=rng,
        init=init,
        n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
        normalize_factors=normalize_factors,
        return_errors=True,
        n_iter_max=100,
        linesearch=linesearch,
    )
    rec_tensor = parafac2_to_tensor(rec)

    error = tl.norm(rec_tensor - tensor, 2)
    error /= tl.norm(tensor, 2)
    assert_(error < tol_norm_2, "norm 2 of reconstruction higher than tol")

    # Test factor correlation
    A_sign = tl.sign(random_parafac2_tensor.factors[0])
    rec_A_sign = tl.sign(rec.factors[0])
    A_corr = congruence_coefficient(
        A_sign * random_parafac2_tensor.factors[0], rec_A_sign * rec.factors[0]
    )[0]
    assert_(A_corr > 0.98)

    C_corr = congruence_coefficient(random_parafac2_tensor.factors[2], rec.factors[2])[
        0
    ]
    assert_(C_corr > 0.98)

    for i, (true_proj, rec_proj) in enumerate(
        zip(random_parafac2_tensor.projections, rec.projections)
    ):
        true_Bi = tl.dot(true_proj, random_parafac2_tensor.factors[1]) * A_sign[i]
        rec_Bi = tl.dot(rec_proj, rec.factors[1]) * rec_A_sign[i]
        Bi_corr = congruence_coefficient(true_Bi, rec_Bi)[0]
        assert_(Bi_corr > 0.98)

    # Test convergence criterion
    noisy_slices = [
        slice_ + tl.tensor(0.001 * rng.standard_normal(tl.shape(slice_)))
        for slice_ in slices
    ]

    rec, err = parafac2(
        noisy_slices,
        rank,
        random_state=rng,
        init=init,
        normalize_factors=normalize_factors,
        tol=1.0e-2,
        return_errors=True,
        linesearch=linesearch,
    )
    assert len(err) > 2  # Check that we didn't just immediately exit
    assert err[-2] - err[-1] < 1.0e-2

    # Check that the previous iteration didn't meet the criteria
    assert err[-3] - err[-2] > 1.0e-2

    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch, parafac2, Parafac2, ignore_args={"return_errors"}, rank=3
    )


def test_parafac2_linesearch():
    """Test that we end up with a better fit at the same number of iterations with linesearch."""
    rng = tl.check_random_state(1234)
    rank = 4

    random_parafac2_tensor = random_parafac2(
        shapes=[(25 + rng.randint(5), 300) for _ in range(15)],
        rank=rank,
        random_state=rng,
    )

    slices = parafac2_to_slices(random_parafac2_tensor)

    _, err = parafac2(
        slices,
        rank,
        init="svd",
        return_errors=True,
        n_iter_max=10,
        linesearch=False,
    )
    standard_error = err[-1]

    _, err = parafac2(
        slices,
        rank,
        init="svd",
        return_errors=True,
        n_iter_max=10,
        linesearch=True,
    )
    ls_error = err[-1]
    assert ls_error < standard_error


def test_linesearch_accepts_only_improved_fit():
    rng = tl.check_random_state(123)
    rank = 4

    weights, factors, projections = random_parafac2(
        shapes=[(25 + rng.randint(5), 300) for _ in range(15)],
        rank=rank,
        random_state=rng,
    )
    slices = parafac2_to_slices((weights, factors, projections))

    # Create dummy variable for the previous iteration
    previous_iteration = random_parafac2(
        shapes=[(25 + rng.randint(5), 300) for _ in range(15)],
        rank=rank,
        random_state=rng,
    )

    # Test with line search where the reconstruction error would worsen if accepted
    line_search = _BroThesisLineSearch(norm_tensor=1, svd="truncated_svd")
    ls_factors, ls_projections, ls_rec_error = line_search.line_step(
        iteration=10,
        tensor_slices=slices,
        factors_last=previous_iteration[1],
        weights=weights,
        factors=factors,
        projections=projections,
        rec_error=0,
    )

    # Assert that the factor matrices, projection and reconstruction error all
    # are unaffected by the line search
    for fm, ls_fm in zip(factors, ls_factors):
        assert_array_almost_equal(fm, ls_fm)
    for proj, ls_proj in zip(projections, ls_projections):
        assert_array_almost_equal(proj, ls_proj)
    assert ls_rec_error == 0
    assert line_search.acc_fail == 1

    # Test with line search where the reconstruction error would improve if accepted
    line_search = _BroThesisLineSearch(norm_tensor=1, svd="truncated_svd")
    ls_factors, ls_projections, ls_rec_error = line_search.line_step(
        iteration=10,
        tensor_slices=slices,
        factors_last=previous_iteration[1],
        weights=weights,
        factors=factors,
        projections=projections,
        rec_error=float("inf"),  # float('inf') to force accepting the line search
    )
    # Assert that the factor matrices, projection and reconstruction error all
    # are changed by the line search
    for fm, ls_fm in zip(factors, ls_factors):
        assert_(tl.norm(fm - ls_fm) > 1e-5)
    for proj, ls_proj in zip(projections, ls_projections):
        assert_(tl.norm(proj - ls_proj) > 1e-5)
    assert 0 < ls_rec_error < float("inf")
    assert line_search.acc_fail == 0


@pytest.mark.parametrize("linesearch", [False, True])
def test_parafac2_nn(linesearch):
    rng = tl.check_random_state(1234)
    tol_norm_2 = 1e-2
    rank = 3

    weights, factors, projections = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(10)],
        rank=rank,
        random_state=rng,
    )
    factors = [tl.abs(factors[0]), factors[1], tl.abs(factors[2])]
    random_parafac2_tensor = Parafac2Tensor((weights, factors, projections))

    # It is difficult to correctly identify B[i, :, r] if A[i, r] is small.
    # This is sensible, since then B[i, :, r] contributes little to the total value of X.
    # To test the PARAFAC2 decomposition in the precence of roundoff errors, we therefore add
    # 0.01 to the A factor matrix.
    random_parafac2_tensor.factors[0] = random_parafac2_tensor.factors[0] + 0.01

    tensor = parafac2_to_tensor(random_parafac2_tensor)
    slices = parafac2_to_slices(random_parafac2_tensor)

    rec, _ = parafac2(
        slices,
        rank,
        random_state=rng,
        init="svd",
        nn_modes=[0, 2],
        n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
        normalize_factors=False,
        return_errors=True,
        n_iter_max=20,
        linesearch=linesearch,
    )
    rec_tensor = parafac2_to_tensor(rec)

    error = tl.norm(rec_tensor - tensor, 2)
    error /= tl.norm(tensor, 2)
    assert_(error < tol_norm_2, "norm 2 of reconstruction higher than tol")

    # Test factor correlation
    A_corr = congruence_coefficient(random_parafac2_tensor.factors[0], rec.factors[0])[
        0
    ]
    assert_(A_corr > 0.98)

    C_corr = congruence_coefficient(random_parafac2_tensor.factors[2], rec.factors[2])[
        0
    ]
    assert_(C_corr > 0.98)

    for i, (true_proj, rec_proj) in enumerate(
        zip(random_parafac2_tensor.projections, rec.projections)
    ):
        true_Bi = tl.dot(true_proj, random_parafac2_tensor.factors[1])
        rec_Bi = tl.dot(rec_proj, rec.factors[1])
        Bi_corr = congruence_coefficient(true_Bi, rec_Bi)[0]
        assert_(Bi_corr > 0.98)

    # Fit with only one iteration to check non-negativity
    weights, factors, projections = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(25)],
        rank=rank,
        random_state=rng,
    )
    # The default random parafac2 tensor has non-negative A and C
    # we therefore multiply them randomly with -1, 0 or 1 to get both positive and negative components
    factors = [
        factors[0] * tl.tensor(rng.randint(-1, 2, factors[0].shape), dtype=tl.float64),
        factors[1],
        factors[2] * tl.tensor(rng.randint(-1, 2, factors[2].shape), dtype=tl.float64),
    ]
    slices = parafac2_to_slices((weights, factors, projections))
    rec, _ = parafac2(
        slices,
        rank,
        random_state=rng,
        init="random",
        nn_modes=[0, 2],
        n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
        normalize_factors=False,
        return_errors=True,
        n_iter_max=1,
        linesearch=linesearch,
    )
    assert_(tl.all(rec[1][0] > -1e-10))
    assert_(tl.all(rec[1][2] > -1e-10))

    # Test that constraining B leads to a warning
    with pytest.warns(UserWarning):
        rec = parafac2(
            slices,
            rank,
            random_state=rng,
            init="random",
            nn_modes=[0, 1, 2],
            n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
            normalize_factors=False,
            n_iter_max=1,
            linesearch=linesearch,
        )
    with pytest.warns(UserWarning):
        rec = parafac2(
            slices,
            rank,
            random_state=rng,
            init="random",
            nn_modes="all",
            n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
            normalize_factors=False,
            n_iter_max=1,
            linesearch=linesearch,
        )


def test_parafac2_slice_and_tensor_input():
    rng = tl.check_random_state(1234)
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(15, 30) for _ in range(25)], rank=rank, random_state=rng
    )
    tensor = parafac2_to_tensor(random_parafac2_tensor)
    slices = parafac2_to_slices(random_parafac2_tensor)

    slice_rec = parafac2(
        slices, rank, init="svd", normalize_factors=False, n_iter_max=2
    )
    slice_rec_tensor = parafac2_to_tensor(slice_rec)

    tensor_rec = parafac2(
        tensor, rank, init="svd", normalize_factors=False, n_iter_max=2
    )
    tensor_rec_tensor = parafac2_to_tensor(tensor_rec)

    assert_array_almost_equal(slice_rec_tensor, tensor_rec_tensor)


def test_parafac2_normalize_factors():
    rng = tl.check_random_state(1234)
    rank = 2  # Rank 2 so we only need to test rank of minimum and maximum

    random_parafac2_tensor = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(25)],
        rank=rank,
        random_state=rng,
    )
    random_parafac2_tensor.factors[0] = random_parafac2_tensor.factors[0] + 0.1
    norms = tl.ones(rank)
    for factor in random_parafac2_tensor.factors:
        norms = norms * tl.norm(factor, axis=0)

    slices = parafac2_to_tensor(random_parafac2_tensor)

    unnormalized_rec = parafac2(
        slices, rank, random_state=rng, normalize_factors=False, n_iter_max=2
    )
    assert unnormalized_rec.weights[0] == 1

    normalized_rec = parafac2(slices, rank, random_state=rng, normalize_factors=True)

    assert_array_almost_equal(tl.norm(normalized_rec.factors[0], axis=0), tl.ones(rank))
    assert_array_almost_equal(tl.norm(normalized_rec.factors[1], axis=0), tl.ones(rank))
    assert_array_almost_equal(tl.norm(normalized_rec.factors[2], axis=0), tl.ones(rank))
    assert tl.abs(tl.max(norms) - tl.max(normalized_rec.weights)) / tl.max(norms) < 0.05
    assert tl.abs(tl.min(norms) - tl.min(normalized_rec.weights)) / tl.min(norms) < 0.05


def test_parafac2_init_valid():
    rng = tl.check_random_state(1234)
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(15, 30)] * 25, rank=rank, random_state=rng
    )
    tensor = parafac2_to_tensor(random_parafac2_tensor)
    weights, (A, B, C), projections = random_parafac2_tensor
    B = tl.dot(projections[0], B)

    for init_method in ["random", "svd", random_parafac2_tensor, (weights, (A, B, C))]:
        init = initialize_decomposition(tensor, rank, init=init_method)
        assert init.shape == random_parafac2_tensor.shape


def test_parafac2_init_cross_product():
    """Test that SVD initialization using the cross-product or concatenated
    tensor yields the same result."""
    rng = tl.check_random_state(1234)
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(25, 100)] * 3, rank=rank, random_state=rng
    )
    slices = parafac2_to_slices(random_parafac2_tensor)

    init = initialize_decomposition(slices, rank, init="svd")

    # Double the number of matrices so that we switch to the cross-product
    init_double = initialize_decomposition(slices + slices, rank, init="svd")

    # These factor matrices should be essentially the same
    assert_allclose(init.factors[1], init_double.factors[1], rtol=1e-3, atol=1e-5)
    assert_allclose(init.factors[2], init_double.factors[2], rtol=1e-3, atol=1e-5)


def test_parafac2_init_error():
    rng = tl.check_random_state(1234)
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(15, 30)] * 25, rank=rank, random_state=rng
    )
    tensor = parafac2_to_tensor(random_parafac2_tensor)

    with pytest.raises(ValueError):
        _ = initialize_decomposition(tensor, rank, init="bogus init type")

    with pytest.raises(ValueError):
        _ = initialize_decomposition(
            tensor, rank, init=("another", "bogus", "init", "type")
        )

    rank = 4
    random_parafac2_tensor = random_parafac2(
        shapes=[(15, 3)] * 25, rank=rank, random_state=rng
    )
    tensor = parafac2_to_tensor(random_parafac2_tensor)

    with pytest.raises(Exception):
        _ = initialize_decomposition(tensor, rank, init="svd")


def test_parafac2_to_tensor():
    rng = tl.check_random_state(1234)
    rank = 3

    I = 25
    J = 15
    K = 30

    weights, factors, projections = random_parafac2(
        shapes=[(J, K)] * I, rank=rank, random_state=rng
    )

    constructed_tensor = parafac2_to_tensor((weights, factors, projections))

    for i in range(I):
        Bi = tl.dot(projections[i], factors[1])
        manual_tensor = tl.einsum("r,jr,kr", factors[0][i], Bi, factors[2])
        assert_(tl.max(tl.abs(constructed_tensor[i, :, :] - manual_tensor)) < 1e-6)
