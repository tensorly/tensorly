from .. import outer
from ... import cp_to_tensor
from ...random import random_cp
from ...testing import assert_array_almost_equal

def test_outer():
    """Test for outer
    """
    rank = 3
    weigths, factors = random_cp((3, 4, 2), rank=rank, normalise_factors=False)
    true_rec = cp_to_tensor((weigths, factors))
    rec = 0
    for r in range(rank):
        rec = rec + outer([f[:, r] for f in factors])
    assert_array_almost_equal(true_rec, rec)
