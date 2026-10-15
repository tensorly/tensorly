import pytest
import numpy as np
import tensorly as tl

from ..leverage_scores import leverage_score_dist
from ...testing import assert_array_almost_equal, assert_equal


@pytest.mark.parametrize("no_row", [100, 200])
@pytest.mark.parametrize("no_col", [20, 80])
@pytest.mark.parametrize("rank", [10, 20])
def test_leverage_score_dist(no_row, no_col, rank):
    # Create no_row-by-no_col matrix of given rank
    U = tl.randn((no_row, rank), dtype=tl.float64)
    U, _ = tl.qr(U, mode="reduced")
    A = U @ tl.randn((rank, no_col), dtype=tl.float64)

    # Compute leverage scores via leverage_score_dist function
    lev_score_dist = leverage_score_dist(A)

    # Assert that all entries of distribution are non-negative
    assert_equal(any(lev_score_dist >= 0), True)

    # Assert that the distribution sums to one
    assert_array_almost_equal(tl.sum(lev_score_dist), 1.0)

    # Assert that lev_score_dist matches true leverage score distribution
    true_lev_score_dist = tl.sum(U**2, axis=1) / rank
    assert_array_almost_equal(lev_score_dist, true_lev_score_dist)


@pytest.mark.parametrize(
    "dtype", [] if tl.get_backend() == "paddle" else ["complex64", "complex128"]
)  # Paddle v3.0 SVD does not support complex inputs.
@pytest.mark.parametrize(
    "matrix",
    [
        [[1j, 0], [0, 1]],
        [[1j, 1], [2, -1j], [1 + 1j, 2j]],
        [[1j, 1, 2], [2, -1j, 1 + 1j]],
        [[1j, 2j], [1, 2], [2j, 4j]],
    ],
    ids=["imaginary-diagonal", "tall", "wide", "rank-deficient"],
)
def test_complex_leverage_score_dist(matrix, dtype):
    matrix = np.asarray(matrix, dtype=dtype)
    tensor = tl.tensor(matrix, dtype=getattr(tl.backend, dtype))
    original = tl.copy(tensor)
    distribution = leverage_score_dist(tensor)
    # The diagonal of the orthogonal projector contains unnormalized scores.
    expected = np.diag(matrix @ np.linalg.pinv(matrix)).real
    expected = expected / np.linalg.matrix_rank(matrix)
    result = tl.to_numpy(distribution)
    assert_equal(tl.context(distribution)["dtype"], tl.float64)
    assert_equal(np.all(np.isfinite(result)), True)
    assert_equal(np.all(result >= 0), True)
    assert_array_almost_equal(result.sum(), 1.0)
    assert_array_almost_equal(result, expected, decimal=5)
    assert_array_almost_equal(tensor, original)
    tl.check_random_state(1234).choice(len(result), size=10, p=result)
