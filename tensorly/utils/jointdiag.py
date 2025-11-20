from itertools import combinations

import tensorly as tl

# Authors: Aaron Meyer <a@ameyer.me>
#          Sebastian Seljak

# License: BSD 3 clause


def deviation_from_normality(matrices_tensor):
    """
    Calculates the total deviation from normality for a set of matrices.

    Metric: Sum of || A @ A.T - A.T @ A ||_F^2 for all matrices in the tensor.

    Args:
        matrices_tensor (Tensor): Dimension (k, k, n)

    Returns:
        float: The total deviation error.
    """
    n_matrices = matrices_tensor.shape[2]
    total_deviation = 0.0

    for i in range(n_matrices):
        A = matrices_tensor[:, :, i]
        # Calculate A Transpose
        A_t = tl.transpose(A)

        # Calculate Commutator: (A * A^T) - (A^T * A)
        # Note: Depending on the backend, tl.dot might act differently on 2D matrices.
        # Using explicit matrix multiplication is safer if available,
        # but here is the standard dot approach for 2D slices:
        commutator = tl.dot(A, A_t) - tl.dot(A_t, A)

        # Add the squared norm of the commutator
        total_deviation += tl.norm(commutator) ** 2

    return total_deviation


def joint_matrix_diagonalization(
    matrices_tensor,
    max_n_iter: int = 50,
    threshold: float = 1e-8,
    verbose: bool = False,
):
    """
    Jointly diagonalizes n matrices, organized in tensor of dimension (k,k,n).
    Returns the diagonalized matrices, along with the transformation matrix [1]_ .

    Notes
    -----
    This algorithm performs joint eigenstructure estimation for a set of
    non-defective matrices by computing a similarity transformation matrix
    ``P`` that simultaneously diagonalizes the set.

    **Algorithm Details**
    The method minimizes the Frobenius norm of the off-diagonal elements of the
    transformed matrices. Unlike methods restricted to simultaneous Schur
    decomposition (which use only unitary transformations), this algorithm
    alternates between two specific updates:

    1. **Non-unitary Shear Transformations:** Reduces the matrices' departure
        from normality.
    2. **Unitary Rotations:** Minimizes the off-diagonal norm of the resulting
        matrices.

    **Scrambling and Robustness**
    This approach is specifically designed to handle "scrambling" defined as
    significant deviation from normality. Purely unitary methods often fail to
    converge when input matrices are far from normal or possess repeated/close
    eigenvalues. By incorporating non-unitary shears, this algorithm robustly
    converges in these difficult scenarios and exhibits an asymptotically
    quadratic convergence rate.

    Args:
        matrices_tensor (_type_): n matrices, organized in a single tensor of dimension (k, k, n).
        max_n_iter (int, optional): Maximum iteration number. Defaults to 50.
        threshold (float, optional): Threshold for decrease in deviation indicating convergence. Defaults to 1e-8.
        verbose (bool, optional): Output progress information during diagonalization. Defaults to False.

    Raises:
        RuntimeError: Error raised if a shear angle cannot be found.

    Returns:
        Tensor: X after joint diagonalization.
        Tensor: The transformation matrix resulting in the diagonalization.

    References
    ----------
    .. [1] T. Fu and X. Gao, “Simultaneous diagonalization with similarity transformation for
       non-defective matrices”, in Proc. IEEE International Conference on Acoustics, Speech
       and Signal Processing (ICASSP 2006), vol. IV, pp. 1137-1140, Toulouse, France, May 2006.
    """
    matrices_tensor = tl.copy(matrices_tensor)
    matrix_dimension = tl.shape(matrices_tensor)[0]  # Dimension of square matrix slices
    assert tl.ndim(matrices_tensor) == 3, "Input must be a 3D tensor"
    assert matrix_dimension == matrices_tensor.shape[1], "All matrices must be square."

    # Deviation from normality is strictly decreasing
    deviation = deviation_from_normality(matrices_tensor)

    if verbose:
        print(f"Sweep # 0: dev = {deviation:.3e}")

    # Initialize transformation matrix as identity
    transform_P = tl.eye(matrix_dimension)

    for k in range(max_n_iter):
        # loop over all pairs of slices
        for p, q in combinations(range(matrix_dimension), 2):
            # Comparing the p and q chords across matrices, identifies the
            # position h with the largest difference
            diag_diff_vector = matrices_tensor[p, p, :] - matrices_tensor[q, q, :]
            h = tl.argmax(tl.abs(diag_diff_vector))

            # List of non-selected indices
            all_but_pq = list(set(range(matrix_dimension)) - set([p, q]))

            # Compute certain quantities
            dh = diag_diff_vector[h]
            matrix_h = matrices_tensor[:, :, h]
            off_diag_shear_mismatch = tl.dot(
                matrix_h[p, all_but_pq], matrix_h[q, all_but_pq]
            ) - tl.dot(matrix_h[all_but_pq, p], matrix_h[all_but_pq, q])
            off_diag_energy = (
                tl.norm(matrix_h[p, all_but_pq]) ** 2
                + tl.norm(matrix_h[q, all_but_pq]) ** 2
                + tl.norm(matrix_h[all_but_pq, p]) ** 2
                + tl.norm(matrix_h[all_but_pq, q]) ** 2
            )
            matrix_h_pq_diff = matrix_h[p, q] - matrix_h[q, p]

            # Build shearing matrix out of these quantities
            shear_angle = tl.arctanh(
                (off_diag_shear_mismatch - matrix_h_pq_diff * dh)
                / (2 * (dh**2 + matrix_h_pq_diff**2) + off_diag_energy)
            )

            # Inverse of Sk on left side
            prev_vec = tl.copy(matrices_tensor[p, :, :])
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[p, :, :],
                matrices_tensor[p, :, :] * tl.cosh(shear_angle)
                - matrices_tensor[q, :, :] * tl.sinh(shear_angle),
            )
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[q, :, :],
                -prev_vec * tl.sinh(shear_angle)
                + matrices_tensor[q, :, :] * tl.cosh(shear_angle),
            )

            # Sk on right side
            prev_vec = tl.copy(matrices_tensor[:, p, :])
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[:, p, :],
                matrices_tensor[:, p, :] * tl.cosh(shear_angle)
                + matrices_tensor[:, q, :] * tl.sinh(shear_angle),
            )
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[:, q, :],
                prev_vec * tl.sinh(shear_angle)
                + matrices_tensor[:, q, :] * tl.cosh(shear_angle),
            )

            # Update transform_P
            prev_vec = tl.copy(transform_P[:, p])
            transform_P = tl.index_update(
                transform_P,
                tl.index[:, p],
                transform_P[:, p] * tl.cosh(shear_angle)
                + transform_P[:, q] * tl.sinh(shear_angle),
            )
            transform_P = tl.index_update(
                transform_P,
                tl.index[:, q],
                prev_vec * tl.sinh(shear_angle)
                + transform_P[:, q] * tl.cosh(shear_angle),
            )

            # Defines array of off-diagonal element differences
            offdiag_symmetry_diff = -matrices_tensor[q, p, :] - matrices_tensor[p, q, :]

            # Compute rotation angle
            Esum = 2 * tl.dot(offdiag_symmetry_diff, diag_diff_vector)
            Dsum = tl.dot(diag_diff_vector, diag_diff_vector) - tl.dot(
                offdiag_symmetry_diff, offdiag_symmetry_diff
            )

            base_rotation_angle = tl.arctan(Esum / Dsum)
            angle_selection = (
                tl.cos(base_rotation_angle) * Dsum + tl.sin(base_rotation_angle) * Esum
            )

            # Defines 1 of 2 possible angles
            if angle_selection > 0.0:
                theta_k = base_rotation_angle / 4
            elif angle_selection < 0.0:
                theta_k = (base_rotation_angle + tl.pi) / 4
            else:
                raise RuntimeError("joint_matrix_diagonalization: No solution found.")

            # Given's rotation, this will minimize norm of off-diagonal elements only
            prev_vec = tl.copy(matrices_tensor[p, :, :])
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[p, :, :],
                matrices_tensor[p, :, :] * tl.cos(theta_k)
                - matrices_tensor[q, :, :] * tl.sin(theta_k),
            )
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[q, :, :],
                prev_vec * tl.sin(theta_k) + matrices_tensor[q, :, :] * tl.cos(theta_k),
            )

            # Right side rotation
            prev_vec = tl.copy(matrices_tensor[:, p, :])
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[:, p, :],
                matrices_tensor[:, p, :] * tl.cos(theta_k)
                - matrices_tensor[:, q, :] * tl.sin(theta_k),
            )
            matrices_tensor = tl.index_update(
                matrices_tensor,
                tl.index[:, q, :],
                prev_vec * tl.sin(theta_k) + matrices_tensor[:, q, :] * tl.cos(theta_k),
            )

            # Update transform_P
            prev_vec = tl.copy(transform_P[:, p])
            transform_P = tl.index_update(
                transform_P,
                tl.index[:, p],
                transform_P[:, p] * tl.cos(theta_k)
                - transform_P[:, q] * tl.sin(theta_k),
            )
            transform_P = tl.index_update(
                transform_P,
                tl.index[:, q],
                prev_vec * tl.sin(theta_k) + transform_P[:, q] * tl.cos(theta_k),
            )

        # Update deviation from normality
        old_deviation = deviation
        deviation = deviation_from_normality(matrices_tensor)

        if verbose:
            print(f"Sweep # {k + 1}: dev = {deviation:.3e}")

        if (old_deviation - deviation < threshold) or (deviation < threshold):
            break

    return matrices_tensor, transform_P
