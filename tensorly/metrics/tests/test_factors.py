import itertools

import numpy as np

import pytest
import tensorly as tl

from ..factors import congruence_coefficient
from tensorly.random import random_cp
from tensorly.cp_tensor import cp_permute_factors


def _tucker_congruence(A, B):
    As = A / tl.norm(A, axis=0)
    Bs = B / tl.norm(B, axis=0)

    return tl.dot(tl.transpose(As), Bs)


def _congruence_coefficient_slow(A, B, absolute_value):
    _, r = tl.shape(A)
    corr_matrix = _tucker_congruence(A, B)
    if absolute_value:
        corr_matrix = tl.abs(corr_matrix)
    corr_matrix = tl.to_numpy(corr_matrix)

    best_corr = None
    best_permutation = None
    for permutation in itertools.permutations(range(r)):
        corr = 0
        for i, j in zip(range(r), permutation):
            corr += corr_matrix[i, j] / r

        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_permutation = permutation

    return best_corr, best_permutation


@pytest.mark.parametrize(
    ("I", "R", "absolute_value"),
    list(
        itertools.product(
            (1, 3, 5, 10, 100),
            (1, 3, 5),
            (True, False),
        )
    ),
)
def test_congruence_coefficient(I, R, absolute_value):
    rng = tl.check_random_state(1234)
    A = tl.tensor(rng.standard_normal((I, R)))
    B = tl.tensor(rng.standard_normal((I, R)))

    fast_congruence, fast_permutation = congruence_coefficient(
        A, B, absolute_value=absolute_value
    )
    slow_congruence, slow_permutation = _congruence_coefficient_slow(
        A, B, absolute_value=absolute_value
    )
    assert fast_congruence == pytest.approx(slow_congruence)
    if I != 1:
        assert fast_permutation == list(slow_permutation)

    # Adding test from @maximeguillaud issue #487
    shape = (3, 4, 5)
    rank = 4
    cp_tensor_1 = random_cp(shape, rank, random_state=0)
    cp_tensor_2 = cp_tensor_1.cp_copy()
    col_order_2 = [3, 1, 2, 0]
    for f in range(3):
        cp_tensor_2.factors[f] = cp_tensor_1.factors[f][:, col_order_2] * (-1) ** (
            f + 1
        )
    _, permutation = cp_permute_factors(cp_tensor_1, cp_tensor_2)

    assert permutation[0].tolist() == col_order_2


@pytest.mark.parametrize("dtype, scale", [("float32", 1e-6), ("float64", 1e-6)])
def test_congruence_nonzero_norm_product_underflow(dtype, scale):
    matrix = tl.tensor(
        np.random.RandomState(42).normal(size=(8, 64)), dtype=getattr(tl.backend, dtype)
    )
    matrix = matrix * scale
    reversed_matrix = matrix[:, list(range(63, -1, -1))]
    score, permutation = congruence_coefficient(matrix, reversed_matrix)
    assert float(score) == pytest.approx(1, abs=1e-6)
    assert permutation == list(range(63, -1, -1))


@pytest.mark.parametrize("which", [0, 1])
def test_congruence_rejects_zero_column_with_small_norms(which):
    matrix = (
        tl.tensor(np.random.RandomState(42).normal(size=(8, 64)), dtype=tl.float32)
        * 1e-6
    )
    zero_column = tl.index_update(matrix, tl.index[:, 0], 0)
    pair = [matrix, matrix]
    pair[which] = zero_column
    with pytest.raises(ValueError, match="nonzero l2 norm"):
        congruence_coefficient(*pair)


@pytest.fixture
def complex_norm_supported(dtype):
    if tl.get_backend() == "paddle":
        probe = tl.tensor([1 + 1j], dtype=getattr(tl, dtype))
        try:
            tl.norm(probe, axis=0)
        except RuntimeError as error:
            message = str(error)
            if "p_norm" in message and "not registered" in message:
                pytest.skip("This Paddle build has no complex p_norm kernel")
            raise


def _complex_orthonormal_columns(n_columns, dtype):
    if n_columns < 3:
        matrix = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
        matrix = matrix[:, :n_columns]
    else:
        indices = np.arange(n_columns)
        matrix = np.exp(2j * np.pi * np.outer(indices, indices) / n_columns)
        matrix = matrix / np.sqrt(n_columns)
    return tl.tensor(matrix, dtype=getattr(tl, dtype))


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n_columns", [1, 2, 3])
@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_complex_congruence_coefficient(
    dtype, n_columns, as_list, permuted, complex_norm_supported
):
    matrix = _complex_orthonormal_columns(n_columns, dtype)
    order = np.roll(np.arange(n_columns), 1) if permuted else np.arange(n_columns)
    phases = (
        np.exp(1j * np.arange(1, n_columns + 1) * 0.37)
        if permuted
        else np.ones(n_columns)
    )
    other = matrix[:, order] * tl.tensor(phases, **tl.context(matrix))
    left = [matrix, 2 * matrix] if as_list else matrix
    right = [other, 3 * other] if as_list else other
    score, permutation = congruence_coefficient(left, right)
    assert score == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert permutation == np.argsort(order).tolist()


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("as_list", [False, True])
def test_cp_permute_complex_factors(dtype, as_list, complex_norm_supported):
    matrix = _complex_orthonormal_columns(2, dtype)
    weights = tl.tensor([1.0, 2.0], **tl.context(matrix))
    reference = tl.cp_tensor.CPTensor((weights, [matrix, tl.copy(matrix)]))
    other = reference.cp_copy()
    order = tl.tensor([1, 0], dtype=tl.int64)
    for mode in range(2):
        other.factors[mode] = other.factors[mode][:, order]
    other.weights = other.weights[order]
    before = other.to_tensor()
    actual, permutations = cp_permute_factors(reference, [other] if as_list else other)
    assert tl.to_numpy(permutations[0]).tolist() == [1, 0]
    assert np.allclose(tl.to_numpy(actual.weights), tl.to_numpy(weights))
    assert np.allclose(tl.to_numpy(actual.to_tensor()), tl.to_numpy(before))
