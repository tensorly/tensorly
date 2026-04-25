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


@pytest.mark.parametrize("shape", [(10, 5), (10, 10), (5, 10)])
@pytest.mark.parametrize("rank", [1, 3, 10])
@pytest.mark.parametrize("is_u_based_flip_sign", [False, True])
@pytest.mark.parametrize(
    "is_complex",
    [False] if tl.get_backend() is "paddle" else [True, False],
)  # due to unsupported svd of cplx input in v3.0
def test_svd_interface_approx(shape, rank, is_complex, is_u_based_flip_sign):
    """Test that SVD interface can approximate input matrix"""
    tol = 1e-6

    rng = tl.check_random_state(1234)
    # Generate left and right matrices
    R = tl.tensor(rng.random_sample((shape[0], rank)))
    L = tl.tensor(rng.random_sample((rank, shape[1])))

    if is_complex:
        R = R + 1j * tl.tensor(rng.random_sample((shape[0], rank)))
        L = L + 1j * tl.tensor(rng.random_sample((rank, shape[1])))

    # Fixed-rank input
    X = R @ L

    U, S, V = svd_interface(
        X, n_eigenvecs=rank, non_negative=False, u_based_flip_sign=is_u_based_flip_sign
    )

    # Check approximation error
    r = min(rank, *shape)
    S_imag = 0j if is_complex else 0  # for dtype casting of S matrix
    X_aprox = U[:, :r] @ tl.diag(S[:r] + S_imag) @ V[:r, :]
    err = tl.norm(X - X_aprox, 2) / tl.norm(X, 2)
    assert_(tl.abs(err) < tol)  # abs due to cplx output of tf
