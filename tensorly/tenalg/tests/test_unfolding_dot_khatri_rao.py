import numpy as np
import tensorly as tl
from ...random import random_cp
from ...base import unfold
from ...testing import assert_array_almost_equal

from .. import khatri_rao
from .. import unfolding_dot_khatri_rao


def test_unfolding_dot_khatri_rao():
    """Test for unfolding_dot_khatri_rao

    Check against other version check sparse safe
    """
    shape = (10, 10, 10, 4)
    rank = 5
    tensor = tl.tensor(np.random.random(shape))
    weights, factors = random_cp(
        shape=shape, rank=rank, full=False, normalise_factors=True
    )

    for mode in range(tl.ndim(tensor)):
        # Version forming explicitely the khatri-rao product
        unfolded = unfold(tensor, mode)
        kr_factors = khatri_rao(factors, weights=weights, skip_matrix=mode)
        true_res = tl.dot(unfolded, kr_factors)

        # Efficient sparse-safe version
        res = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)
        assert_array_almost_equal(true_res, res, decimal=3)
