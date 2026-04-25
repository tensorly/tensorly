import pytest
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
