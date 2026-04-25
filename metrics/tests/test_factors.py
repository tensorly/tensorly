import itertools

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
    itertools.product(
        (1, 3, 5, 10, 100),
        (1, 3, 5),
        (True, False),
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
