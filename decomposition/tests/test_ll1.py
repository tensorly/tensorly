"""
Tests for the LL1 decomposition module.

* Unconstrained LL1 via ALS: ground-truth noiseless data, tolerance 1e-9.
* Stokes-constrained LL1 via BPG and AO-ADMM: noisy data, tolerance 1e-5.
"""

import pytest
import numpy as np

import tensorly as tl
from .._ll1 import ll1_als, LL1
from .._ll1_constrained import ll1_bpg, ll1_ao_admm, LL1_BPG, LL1_AO_ADMM
from ...ll1_tensor import (
    LL1Tensor,
    ll1_to_tensor,
    _validate_ll1_tensor,
    check_ll1_uniqueness,
)
from ...datasets.ll1_synthetic import gen_ll1


# ---------------------------------------------------------------------------
# LL1Tensor class tests
# ---------------------------------------------------------------------------


def test_ll1tensor_validation():
    """LL1Tensor validates (A, B, C) structure correctly."""
    rng = np.random.RandomState(42)
    R, L, I, J, K = 2, 3, 5, 6, 4
    A = tl.tensor(rng.random_sample((I, L * R)))
    B = tl.tensor(rng.random_sample((J, L * R)))
    C = tl.tensor(rng.random_sample((K, R)))

    ll1 = LL1Tensor((A, B, C))
    assert ll1.shape == (I, J, K)
    assert ll1.rank == R
    assert ll1.column_rank == L

    # Mismatched column counts
    B_bad = tl.tensor(rng.random_sample((J, L * R + 1)))
    with pytest.raises(ValueError):
        LL1Tensor((A, B_bad, C))

    # Indivisible column count (A has 6 cols; C with 4 cols -> 6 % 4 != 0)
    C_bad = tl.tensor(rng.random_sample((K, 4)))
    with pytest.raises(ValueError):
        LL1Tensor((A, B, C_bad))

    # Zero rank
    C_zero = tl.tensor(rng.random_sample((K, 0)).reshape(K, 0))
    with pytest.raises(ValueError):
        LL1Tensor(
            (
                tl.tensor(rng.random_sample((I, 0)).reshape(I, 0)),
                tl.tensor(rng.random_sample((J, 0)).reshape(J, 0)),
                C_zero,
            )
        )


def test_ll1tensor_roundtrip():
    """to_tensor reconstructs correctly."""
    rng = np.random.RandomState(7)
    R, L, I, J, K = 2, 3, 4, 5, 6
    A = tl.tensor(rng.random_sample((I, L * R)))
    B = tl.tensor(rng.random_sample((J, L * R)))
    C = tl.tensor(rng.random_sample((K, R)))

    ll1 = LL1Tensor((A, B, C))
    tensor = ll1.to_tensor()
    assert tl.shape(tensor) == (I, J, K)

    # Manual reconstruction
    manual = tl.zeros((I, J, K))
    for r in range(R):
        A_r = A[:, r * L : (r + 1) * L]
        B_r = B[:, r * L : (r + 1) * L]
        c_r = C[:, r]
        M_r = tl.dot(A_r, tl.transpose(B_r))
        manual = manual + tl.reshape(M_r, (I, J, 1)) * tl.reshape(c_r, (1, 1, K))

    np.testing.assert_allclose(
        tl.to_numpy(tensor), tl.to_numpy(manual), atol=1e-12
    )


def test_ll1tensor_getitem_setitem():
    """Index access and assignment on (A, B, C)."""
    rng = np.random.RandomState(0)
    A = tl.tensor(rng.random_sample((3, 4)))
    B = tl.tensor(rng.random_sample((5, 4)))
    C = tl.tensor(rng.random_sample((6, 2)))

    ll1 = LL1Tensor((A, B, C))

    assert ll1[0] is ll1.A
    assert ll1[1] is ll1.B
    assert ll1[2] is ll1.C

    new_A = tl.tensor(rng.random_sample((3, 4)))
    ll1[0] = new_A
    assert ll1.A is new_A

    with pytest.raises(IndexError):
        _ = ll1[3]


def test_ll1tensor_iter_len():
    """Iteration yields (A, B, C) and len is 3."""
    rng = np.random.RandomState(1)
    A = tl.tensor(rng.random_sample((3, 2)))
    B = tl.tensor(rng.random_sample((4, 2)))
    C = tl.tensor(rng.random_sample((5, 1)))

    ll1 = LL1Tensor((A, B, C))
    parts = list(ll1)
    assert len(parts) == 3
    assert len(ll1) == 3


# ---------------------------------------------------------------------------
# Uniqueness check
# ---------------------------------------------------------------------------


def test_check_ll1_uniqueness():
    """Uniqueness sufficient condition returns expected results."""
    # R=2, L=2, I=10, J=10, K=5: L*(R-1)+1 = 3 <= 10, R=2 <= K=5 -> True
    assert check_ll1_uniqueness((10, 10, 5), rank=2, column_rank=2) is True

    # R exceeds K -> False
    assert check_ll1_uniqueness((10, 10, 2), rank=3, column_rank=1) is False

    # L too large for I -> False
    assert check_ll1_uniqueness((3, 10, 10), rank=2, column_rank=4) is False


def test_check_ll1_uniqueness_shape_error():
    """Raises on non-third-order shape."""
    with pytest.raises(ValueError):
        check_ll1_uniqueness((10, 10), rank=1, column_rank=1)


# ---------------------------------------------------------------------------
# Unconstrained LL1 ALS  (tolerance 1e-9 on noiseless data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank, column_rank",
    [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)],
)
def test_ll1_als_noiseless(rank, column_rank):
    """ALS recovers an exact LL1 tensor with reconstruction error < 1e-9."""
    # Ensure I, J are large relative to L*R for well-conditioned normal equations
    LR = column_rank * rank
    shape = (max(4 * LR, 8), max(4 * LR + 1, 9), max(rank + 3, 6))
    tensor, ll1_true = gen_ll1(
        shape, rank, column_rank, noise_level=0.0, random_state=42
    )

    ll1_est, errors = ll1_als(
        tensor,
        rank=rank,
        column_rank=column_rank,
        n_iter_max=1000,
        init="random",
        tol=1e-12,
        random_state=12,
        return_errors=True,
    )

    rec_error = tl.norm(tensor - ll1_to_tensor(ll1_est), 2)
    assert float(tl.to_numpy(rec_error)) < 1e-9, (
        f"Reconstruction error {float(tl.to_numpy(rec_error))} exceeds 1e-9."
    )

    # Errors should be non-increasing (up to float noise)
    for i in range(1, len(errors)):
        assert float(tl.to_numpy(errors[i])) <= (
            float(tl.to_numpy(errors[i - 1])) + 1e-10
        )


def test_ll1_als_return_type():
    """ll1_als return types."""
    shape = (6, 7, 5)
    tensor, _ = gen_ll1(shape, rank=2, column_rank=2, noise_level=0.0, random_state=0)

    result = ll1_als(
        tensor, rank=2, column_rank=2, n_iter_max=10, return_errors=False, random_state=0
    )
    assert isinstance(result, LL1Tensor)

    result2, errs = ll1_als(
        tensor, rank=2, column_rank=2, n_iter_max=10, return_errors=True, random_state=0
    )
    assert isinstance(result2, LL1Tensor)
    assert isinstance(errs, list)
    assert len(errs) > 0


def test_ll1_als_non_3d():
    """ALS raises on non-third-order input."""
    tensor = tl.tensor(np.random.random((4, 5)))
    with pytest.raises(ValueError):
        ll1_als(tensor, rank=1, column_rank=1)


# ---------------------------------------------------------------------------
# LL1 class wrapper
# ---------------------------------------------------------------------------


def test_ll1_class():
    """Class-based interface matches function-based."""
    shape = (12, 13, 6)
    tensor, _ = gen_ll1(shape, rank=2, column_rank=2, noise_level=0.0, random_state=99)

    model = LL1(rank=2, column_rank=2, n_iter_max=1000, tol=1e-12, random_state=10)
    ll1_est = model.fit_transform(tensor)

    assert isinstance(ll1_est, LL1Tensor)
    assert hasattr(model, "errors_")

    rec_error = tl.norm(tensor - ll1_to_tensor(ll1_est), 2)
    assert float(tl.to_numpy(rec_error)) < 1e-9


def test_ll1_class_fit():
    """fit() returns self and stores decomposition_."""
    shape = (5, 6, 4)
    tensor, _ = gen_ll1(shape, rank=2, column_rank=1, noise_level=0.0, random_state=7)

    model = LL1(rank=2, column_rank=1, n_iter_max=100, random_state=0)
    ret = model.fit(tensor)
    assert ret is model
    assert hasattr(model, "decomposition_")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def test_gen_ll1_shape_and_rank():
    """gen_ll1 returns correct shapes and metadata."""
    shape = (10, 12, 5)
    rank, L = 3, 2
    tensor, ll1_gt = gen_ll1(shape, rank, L, noise_level=0.0, random_state=0)

    assert tl.shape(tensor) == shape
    assert ll1_gt.rank == rank
    assert ll1_gt.column_rank == L
    assert ll1_gt.shape == shape
    assert tl.shape(ll1_gt.A) == (10, L * rank)
    assert tl.shape(ll1_gt.B) == (12, L * rank)
    assert tl.shape(ll1_gt.C) == (5, rank)


def test_gen_ll1_stokes_valid():
    """Stokes-constrained generation produces valid factors."""
    shape = (8, 8, 4)
    rank, L = 2, 2
    _, ll1_gt = gen_ll1(shape, rank, L, stokes=True, random_state=42)

    # C columns satisfy Stokes constraint
    C_np = tl.to_numpy(ll1_gt.C)
    for r in range(rank):
        s = C_np[:, r]
        polaris = np.sqrt(s[1] ** 2 + s[2] ** 2 + s[3] ** 2)
        assert s[0] >= 0, "s0 must be non-negative"
        assert s[0] ** 2 >= polaris ** 2 - 1e-12

    # A and B are non-negative
    assert float(tl.to_numpy(tl.min(ll1_gt.A))) >= 0.0
    assert float(tl.to_numpy(tl.min(ll1_gt.B))) >= 0.0


def test_gen_ll1_stokes_wrong_K():
    """gen_ll1 raises when stokes=True and K != 4."""
    with pytest.raises(ValueError):
        gen_ll1((5, 5, 3), rank=1, column_rank=1, stokes=True, random_state=0)


def test_gen_ll1_noise():
    """Adding noise increases reconstruction error against ground truth."""
    shape = (8, 9, 4)
    tensor_clean, _ = gen_ll1(shape, rank=2, column_rank=2, noise_level=0.0, random_state=0)
    tensor_noisy, ll1_gt = gen_ll1(
        shape, rank=2, column_rank=2, noise_level=0.1, random_state=0
    )

    err_clean = float(tl.to_numpy(tl.norm(tensor_clean - ll1_to_tensor(ll1_gt), 2)))
    err_noisy = float(tl.to_numpy(tl.norm(tensor_noisy - ll1_to_tensor(ll1_gt), 2)))
    assert err_noisy > err_clean


# ---------------------------------------------------------------------------
# Constrained LL1 BPG (tolerance 1e-5 on noisy data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2])
def test_ll1_bpg_noisy(rank):
    """BPG recovers a Stokes-constrained LL1 with factor error < 1e-5."""
    L = 2
    shape = (10, 10, 4)
    tensor, ll1_true = gen_ll1(
        shape, rank, L, noise_level=1e-7, stokes=True, random_state=42
    )

    ll1_est, errors = ll1_bpg(
        tensor,
        rank=rank,
        column_rank=L,
        n_iter_max=3000,
        tol=1e-14,
        random_state=7,
        return_errors=True,
    )

    # ||T_est - T_true|| / ||T_true|| < 1e-5
    true_tensor = ll1_to_tensor(ll1_true)
    est_tensor = ll1_to_tensor(ll1_est)
    rel_err = float(
        tl.to_numpy(tl.norm(est_tensor - true_tensor, 2) / tl.norm(true_tensor, 2))
    )
    assert rel_err < 1e-5, f"Relative factor error {rel_err} exceeds 1e-5."

    # Constraints satisfied
    assert float(tl.to_numpy(tl.min(ll1_est.A))) >= -1e-10
    assert float(tl.to_numpy(tl.min(ll1_est.B))) >= -1e-10
    C_np = tl.to_numpy(ll1_est.C)
    for r in range(rank):
        s = C_np[:, r]
        polaris = np.sqrt(s[1] ** 2 + s[2] ** 2 + s[3] ** 2)
        assert s[0] >= -1e-10
        assert s[0] ** 2 >= polaris ** 2 - 1e-8


def test_ll1_bpg_non_3d():
    """BPG raises on non-third-order or wrong K."""
    with pytest.raises(ValueError):
        ll1_bpg(tl.tensor(np.random.random((4, 5))), rank=1, column_rank=1)

    with pytest.raises(ValueError):
        ll1_bpg(tl.tensor(np.random.random((4, 5, 3))), rank=1, column_rank=1)


# ---------------------------------------------------------------------------
# Constrained LL1 AO-ADMM (tolerance 1e-5 on noisy data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2])
def test_ll1_ao_admm_noisy(rank):
    """AO-ADMM recovers a Stokes-constrained LL1 with factor error < 1e-5."""
    L = 2
    shape = (10, 10, 4)
    tensor, ll1_true = gen_ll1(
        shape, rank, L, noise_level=1e-7, stokes=True, random_state=42
    )

    ll1_est, errors = ll1_ao_admm(
        tensor,
        rank=rank,
        column_rank=L,
        n_iter_max=3000,
        n_admm_iter=20,
        rho=1.0,
        tol=1e-14,
        random_state=7,
        return_errors=True,
    )

    true_tensor = ll1_to_tensor(ll1_true)
    est_tensor = ll1_to_tensor(ll1_est)
    rel_err = float(
        tl.to_numpy(tl.norm(est_tensor - true_tensor, 2) / tl.norm(true_tensor, 2))
    )
    assert rel_err < 1e-5, f"Relative factor error {rel_err} exceeds 1e-5."

    # Constraints satisfied
    assert float(tl.to_numpy(tl.min(ll1_est.A))) >= -1e-10
    assert float(tl.to_numpy(tl.min(ll1_est.B))) >= -1e-10
    C_np = tl.to_numpy(ll1_est.C)
    for r in range(rank):
        s = C_np[:, r]
        polaris = np.sqrt(s[1] ** 2 + s[2] ** 2 + s[3] ** 2)
        assert s[0] >= -1e-10
        assert s[0] ** 2 >= polaris ** 2 - 1e-8


def test_ll1_ao_admm_non_3d():
    """AO-ADMM raises on non-third-order or wrong K."""
    with pytest.raises(ValueError):
        ll1_ao_admm(tl.tensor(np.random.random((4, 5))), rank=1, column_rank=1)

    with pytest.raises(ValueError):
        ll1_ao_admm(tl.tensor(np.random.random((4, 5, 3))), rank=1, column_rank=1)


# ---------------------------------------------------------------------------
# Class wrappers for constrained algorithms
# ---------------------------------------------------------------------------


def test_ll1_bpg_class():
    """LL1_BPG class interface."""
    shape = (8, 8, 4)
    tensor, _ = gen_ll1(
        shape, rank=2, column_rank=2, noise_level=0.01, stokes=True, random_state=0
    )

    model = LL1_BPG(rank=2, column_rank=2, n_iter_max=300, random_state=5)
    ll1_est = model.fit_transform(tensor)

    assert isinstance(ll1_est, LL1Tensor)
    assert hasattr(model, "errors_")
    assert hasattr(model, "decomposition_")

    ret = model.fit(tensor)
    assert ret is model


def test_ll1_ao_admm_class():
    """LL1_AO_ADMM class interface."""
    shape = (8, 8, 4)
    tensor, _ = gen_ll1(
        shape, rank=2, column_rank=2, noise_level=0.01, stokes=True, random_state=0
    )

    model = LL1_AO_ADMM(
        rank=2, column_rank=2, n_iter_max=300, n_admm_iter=10, rho=1.0, random_state=5
    )
    ll1_est = model.fit_transform(tensor)

    assert isinstance(ll1_est, LL1Tensor)
    assert hasattr(model, "errors_")

    ret = model.fit(tensor)
    assert ret is model
