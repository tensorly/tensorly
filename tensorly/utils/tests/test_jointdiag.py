import numpy as np

import tensorly as tl

from ..jointdiag import joint_matrix_diagonalization


def test_joint_matrix_diagonalization():
    """
    Generates random diagonal tensor, and random mixing matrix
    Multiplies every slice of diagonal tensor with matrix 'synthetic' S * D * S^-1
    Sends altered tensor into joint_matrix_diagonalization function
    Returns the diagonal tensor estimate and mixing matrix estimate
    """
    k = 14
    d = 10

    rng = np.random.RandomState(1)
    mixing = rng.randn(d, d)
    diags = np.zeros((d, d, k))
    synthetic = np.zeros((d, d, k))

    # Generates k random diagonal matrices and scrambles according to the mixing matrix
    for i in range(k):
        temp_diag = np.diag(rng.randn(d))
        diags[:, :, i] = temp_diag
        synthetic[:, :, i] = np.linalg.inv(mixing) @ temp_diag @ mixing

    synthetic = tl.tensor(synthetic, dtype=tl.float64)

    diag_est, _ = joint_matrix_diagonalization(synthetic, verbose=False)
    diag_est = tl.to_numpy(diag_est)

    ## Sorts outputted diagonal data
    idx_est = np.argsort(np.diag(diag_est[:, :, 0]))
    idx = np.argsort(np.diag(diags[:, :, 0]))
    diag_est = diag_est[idx_est, idx_est, :]
    diags = diags[idx, idx, :]

    # Test that the diagonalized matrices are close to the starting diagonal matrices
    np.testing.assert_allclose(diags, diag_est, atol=1e-7, rtol=1e-7)
