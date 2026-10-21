import pytest
import numpy as np
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
    [False] if tl.get_backend() == "paddle" else [True, False],
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


@pytest.mark.parametrize("shape", [(10, 5), (5, 10)])
@pytest.mark.parametrize("method", ["truncated_svd", "symeig_svd", "randomized_svd"])
def test_svd_interface_caps_n_eigenvecs_at_min_dim(shape, method):
    """Requesting more than min(shape) vectors warns and returns min(shape)."""
    rng = tl.check_random_state(1234)
    X = tl.tensor(rng.random_sample(shape))
    k = min(shape)

    with pytest.warns(UserWarning, match="n_eigenvecs"):
        U, S, V = svd_interface(X, method=method, n_eigenvecs=max(shape))

    assert tl.shape(U) == (shape[0], k)
    assert tl.shape(S) == (k,)
    assert tl.shape(V) == (k, shape[1])


@pytest.mark.parametrize("shape", [(10, 5), (5, 10)])
@pytest.mark.parametrize("method", ["truncated_svd", "symeig_svd", "randomized_svd"])
def test_svd_interface_default_n_eigenvecs(shape, method):
    """Without n_eigenvecs, the reduced SVD is returned without a warning."""
    rng = tl.check_random_state(1234)
    X = tl.tensor(rng.random_sample(shape))
    k = min(shape)

    U, S, V = svd_interface(X, method=method)

    assert tl.shape(U) == (shape[0], k)
    assert tl.shape(S) == (k,)
    assert tl.shape(V) == (k, shape[1])


@pytest.mark.parametrize("shape", [(9, 5), (5, 9), (5, 5)])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_symeig_svd_complex(shape, dtype):
    rng = np.random.RandomState(1234)
    matrix = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    matrix = matrix.astype(dtype)
    X = tl.tensor(matrix, dtype=getattr(tl.backend, dtype))
    U, S, V = svd_interface(X, method="symeig_svd", flip_sign=False)
    U, S, V = [tl.to_numpy(value) for value in (U, S, V)]
    # Forming a Gram matrix squares the condition number; allow its
    # single-precision roundoff in the derived singular vectors.
    tolerance = 5e-5 if dtype == "complex64" else 1e-12
    np.testing.assert_allclose(
        S, np.linalg.svd(matrix, compute_uv=False), rtol=tolerance
    )
    np.testing.assert_allclose((U * S) @ V, matrix, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(min(shape)), atol=tolerance)
    np.testing.assert_allclose(V @ V.conj().T, np.eye(min(shape)), atol=tolerance)


@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (4, 3)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("non_negative", ["nndsvd", "nndsvda"])
def test_nonnegative_svd_zero_direction(shape, dtype, non_negative):
    """Rank-deficient nonnegative inputs have finite NNDSVD initial factors."""
    matrix = np.zeros(shape)
    matrix[0, 0], matrix[1, 1] = 3, 2
    X = tl.tensor(matrix, dtype=getattr(tl.backend, dtype))
    U, _, V = svd_interface(
        X, method="symeig_svd", n_eigenvecs=3, non_negative=non_negative
    )
    expected_U, expected_V = np.zeros((shape[0], 3)), np.zeros((3, shape[1]))
    for i, value in enumerate([3, 2]):
        expected_U[i, i] = expected_V[i, i] = np.sqrt(value)
    if non_negative == "nndsvda":
        expected_U[expected_U == 0] = matrix.mean()
        expected_V[expected_V == 0] = matrix.mean()
    np.testing.assert_allclose(tl.to_numpy(U), expected_U, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tl.to_numpy(V), expected_V, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (4, 3)])
def test_nonnegative_parafac_rank_deficient_svd_init(shape):
    """Native nonnegative CP fitting stays finite when SVD initialization has null directions."""
    from tensorly.decomposition import non_negative_parafac

    matrix = np.zeros(shape)
    matrix[0, 0], matrix[1, 1] = 3, 2
    X = tl.tensor(matrix)
    factors = non_negative_parafac(
        X, rank=3, svd="symeig_svd", n_iter_max=3, random_state=0
    )
    assert np.isfinite(tl.to_numpy(tl.cp_to_tensor(factors))).all()
