import pytest
from ...testing import assert_
from ..svd import svd_interface
import tensorly as tl
from ...testing import assert_


@pytest.mark.parametrize("shape", [(10, 5), (10, 10), (5, 10)])
@pytest.mark.parametrize("rank", [1, 3, 5, 8, 10])
@pytest.mark.parametrize("nn", [False, True])
def test_svd_interface_shape(shape, rank, nn):
    """Test that the SVD interface handles shape edge cases."""
    rng = tl.check_random_state(1234)

    X = tl.tensor(rng.random_sample(shape))
    X_mask = tl.zeros_like(X)
    X_mask = tl.index_update(X_mask, tl.index[1, 3], 1.0)

    U, S, V = svd_interface(
        X,
        n_eigenvecs=rank,
        non_negative=nn,
        mask=X_mask,
        n_iter_mask_imputation=3,
    )
    assert_(tl.all(S >= 0.0))

    if nn:
        assert_(tl.all(U >= 0.0))
        assert_(tl.all(V >= 0.0))
