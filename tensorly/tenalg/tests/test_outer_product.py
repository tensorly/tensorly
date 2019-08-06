from ..outer_product import outer
from ...kruskal_tensor import kruskal_to_tensor
from ...random import random_kruskal
from ...testing import assert_array_almost_equal

def test_outer():
    """Test for outer
    """
    rank = 3
    weigths, factors = random_kruskal((3, 4, 2), rank=rank, normalise_factors=False)
    true_rec = kruskal_to_tensor((weigths, factors))
    rec = 0
    for r in range(rank):
        rec = rec + outer([f[:, r] for f in factors])
    assert_array_almost_equal(true_rec, rec)
