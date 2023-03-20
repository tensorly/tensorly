from sys import version_info

import pytest
import tensorly as tl

from .._prod import _prod


@pytest.mark.skipif(
    version_info[1] < 8,
    reason="prod() not implemented before Python v3.8.",
)
def test_prod():
    """Test for _prod (when math.prod unavailable)"""
    from math import prod

    assert _prod([1, 2, 3, 4]) == prod([1, 2, 3, 4])
    assert _prod([]) == 1
    assert _prod(tl.arange(1, 5)) == _prod([1, 2, 3, 4])
