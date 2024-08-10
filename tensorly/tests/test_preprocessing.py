import pytest
from pytest import approx

import numpy as np
import tensorly as tl

from ..parafac2_tensor import parafac2_to_slices
from ..random import random_parafac2
from ..testing import assert_allclose
from ..decomposition._parafac2 import parafac2, _parafac2_reconstruction_error
from ..preprocessing import svd_compress_tensor_slices, svd_decompress_parafac2_tensor


@pytest.mark.parametrize("normalize_factors", [True, False])
@pytest.mark.parametrize("compression_threshold", [0, 1e-5])
def test_svd_compression_gets_correct_components(
    compression_threshold, normalize_factors
):
    """The compressed data can be used to find the components"""
    rng = tl.check_random_state(1234)
    rank = 3
    tol_norm_2 = 1e-2

    random_parafac2_tensor = random_parafac2(
        shapes=[(20 + rng.randint(5), 19) for _ in range(10)],
        rank=rank,
        random_state=rng,
        dtype=tl.float64,
    )

    slices = parafac2_to_slices(random_parafac2_tensor)
    compressed_slices, loadings = svd_compress_tensor_slices(
        slices, compression_threshold=compression_threshold
    )
    rec_pf2 = parafac2(
        compressed_slices,
        rank,
        random_state=rng,
        normalize_factors=normalize_factors,
        n_iter_max=100,
    )

    decompressed_pf2 = svd_decompress_parafac2_tensor(rec_pf2, loadings)
    rec_slices = parafac2_to_slices(decompressed_pf2)
    for slice, rec_slice in zip(slices, rec_slices):
        slice_error = tl.norm(slice - rec_slice) / tl.norm(slice)
        assert slice_error < tol_norm_2, slice_error


@pytest.mark.parametrize("normalize_factors", [True, False])
def test_svd_compression_doesnt_disturb_error(normalize_factors):
    """The SSE for the compressed data is equal to the SSE of the uncompressed data"""
    rng = tl.check_random_state(1234)
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(20 + rng.randint(5), 19) for _ in range(10)],
        rank=rank,
        random_state=rng,
        dtype=tl.float64,
    )

    slices = parafac2_to_slices(random_parafac2_tensor)
    compressed_slices, loadings = svd_compress_tensor_slices(
        slices, compression_threshold=0
    )
    rec_pf2 = parafac2(
        compressed_slices,
        rank,
        random_state=rng,
        normalize_factors=normalize_factors,
        n_iter_max=100,
    )

    decompressed_pf2 = svd_decompress_parafac2_tensor(rec_pf2, loadings)
    compressed_error = _parafac2_reconstruction_error(compressed_slices, rec_pf2)
    decompressed_error = _parafac2_reconstruction_error(slices, decompressed_pf2)
    assert tl.to_numpy(compressed_error) == approx(tl.to_numpy(decompressed_error))


def test_svd_compression_compresses_only_when_necessary():
    """Compression is only done when the number of rows is less than the number of columns and compression_threshold != 0"""
    rng = tl.check_random_state(1234)
    tensor_slices = [
        tl.tensor(rng.standard_normal((25, 10)), dtype=tl.float64),
        tl.tensor(rng.standard_normal((10, 10)), dtype=tl.float64),
    ]
    compressed_slices, loadings = svd_compress_tensor_slices(tensor_slices)

    # First slice is compressed
    assert compressed_slices[0] is not tensor_slices[0]
    assert tuple(compressed_slices[0].shape) == (10, 10)
    assert tuple(loadings[0].shape) == (25, 10)

    # Second slice is not compressed
    assert compressed_slices[1] is tensor_slices[1]
    assert loadings[1] is None


def test_svd_compression_compresses_uses_compression_threshold():
    """If compression threshold is set, small singular values are truncated"""
    rng = tl.check_random_state(1234)
    slice = tl.tensor(rng.standard_normal((10, 5)), dtype=tl.float64)
    slice = tl.concatenate([slice, slice[:, -1:] + 1e-10], axis=1)

    compressed_slices, loadings = svd_compress_tensor_slices(
        [slice], compression_threshold=1e-5
    )

    assert tl.shape(compressed_slices[0]) == (5, 6)
    assert tl.shape(loadings[0]) == (10, 5)


def test_svd_compression_gives_orthogonal_scores():
    """Compressed tensor slices should have orthogonal rows"""
    rng = tl.check_random_state(1234)
    tensor_slices = [
        tl.tensor(rng.standard_normal((25, 10)), dtype=tl.float64),
        tl.tensor(rng.standard_normal((25, 10)), dtype=tl.float64),
    ]
    compressed_slices, _loadings = svd_compress_tensor_slices(tensor_slices)
    for compressed_slice in compressed_slices:
        gramian = tl.to_numpy(
            tl.matmul(compressed_slice, tl.transpose(compressed_slice))
        )
        diagonal_entries = np.diag(gramian)

        np.testing.assert_allclose(np.diag(diagonal_entries), gramian, atol=1e-10)


def test_svd_compression_gives_orthonormal_loadings():
    """Loading matrices should have orthonormal columns"""
    rng = tl.check_random_state(1234)
    tensor_slices = [
        tl.tensor(rng.standard_normal((25, 10)), dtype=tl.float64),
        tl.tensor(rng.standard_normal((25, 10)), dtype=tl.float64),
    ]
    _compressed_slices, loadings = svd_compress_tensor_slices(tensor_slices)

    for loading_matrix in loadings:
        gramian = tl.matmul(tl.transpose(loading_matrix), loading_matrix)
        assert_allclose(gramian, tl.eye(tl.shape(gramian)[0]), atol=1e-10)
