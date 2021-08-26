from .._prod import _prod
from math import prod
import tensorly as tl

def test_prod():
    """Test for _prod (when math.prod unavailable)"""
    assert _prod([1, 2, 3, 4]) == prod([1, 2, 3, 4])
    assert _prod([]) == 1
    assert _prod(tl.arange(1, 5)) == _prod([1, 2, 3, 4])