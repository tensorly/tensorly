import itertools

import pytest
import tensorly as tl

from ..factors import congruence_coefficient


def _tucker_congruence(A, B):
    As = A/tl.norm(A, axis=0)
    Bs = B/tl.norm(B, axis=0)

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
     itertools.product((1, 3, 5, 10, 100), (1, 3, 5), (True, False),)
)
def test_congruence_coefficient(I, R, absolute_value):
    rng = tl.check_random_state(1234)
    A = tl.tensor(rng.standard_normal((I, R)))
    B = tl.tensor(rng.standard_normal((I, R)))

    fast_congruence, fast_permutation = congruence_coefficient(A, B, absolute_value=absolute_value)
    slow_congruence, slow_permutation = _congruence_coefficient_slow(A, B, absolute_value=absolute_value)
    assert fast_congruence == pytest.approx(slow_congruence)
    if I != 1:
        assert fast_permutation == list(slow_permutation)
