from ..base import check_random_state
import numpy as np
from numpy.testing import assert_raises


def test_check_random_state():
    """Test for check_random_state"""

    # Generate a random state for me
    rns = check_random_state(seed=None)
    assert(isinstance(rns, np.random.RandomState))

    # random state from integer seed
    rns = check_random_state(seed=10)
    assert(isinstance(rns, np.random.RandomState))

    # if it is already a random state, just return it
    cpy_rns = check_random_state(seed=rns)
    assert(cpy_rns is rns)

    # only takes as seed a random state, an int or None
    assert_raises(ValueError, check_random_state, seed='bs')
