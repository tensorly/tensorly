from itertools import combinations
import tensorly as tl

# Authors: Aaron Meyer <a@ameyer.me>
#          Sebastian Seljak

# License: BSD 3 clause


def jointdiag(
    X,
    max_iter: int = 50,
    threshold: float = 1e-10,
    verbose: bool = False,
):
    """
    Jointly diagonalizes n matrices, organized in tensor of dimension (k,k,n).
    Returns the diagonalized matrices, along with the transformation matrix.

    T. Fu and X. Gao, “Simultaneous diagonalization with similarity transformation for
    non-defective matrices”, in Proc. IEEE International Conference on Acoustics, Speech
    and Signal Processing (ICASSP 2006), vol. IV, pp. 1137-1140, Toulouse, France, May 2006.

    Args:
        X (_type_): n matrices, organized in tensor of dimension (k,k,n), for joint diagonalization.
        max_iter (int, optional): Maximum iteration number. Defaults to 50.
        threshold (float, optional): Threshold for decrease in error indicating convergence. Defaults to 1e-10.
        verbose (bool, optional): Output progress information during diagonalization. Defaults to False.

    Raises:
        RuntimeError: Error raised if a shear angle cannot be found.

    Returns:
        Tensor: X after joint diagonalization.
        Tensor: The transformation matrix resulting in the diagonalization.
    """
    X = tl.tensor(X, **tl.context(X))
    D = tl.shape(X)[0]  # Dimension of square matrix slices
    assert tl.ndim(X) == 3, "Input must be a 3D tensor"
    assert D == X.shape[1], "All slices must be square"

    # Initial error calculation
    # Transpose is because np.tril operates on the last two dimensions
    e = tl.norm(X) ** 2.0 - tl.norm(tl.diagonal(X, axis1=1, axis2=2)) ** 2.0

    if verbose:
        print(f"Sweep # 0: e = {e:.3e}")

    # Additional output parameters
    Q_total = tl.eye(D)

    for k in range(max_iter):
        # loop over all pairs of slices
        for p, q in combinations(range(D), 2):
            # Finds matrix slice with greatest variability among diagonal elements
            d_ = X[p, p, :] - X[q, q, :]
            h = tl.argmax(tl.abs(d_))

            # List of indices
            all_but_pq = list(set(range(D)) - set([p, q]))

            # Compute certain quantities
            dh = d_[h]
            Xh = X[:, :, h]
            Kh = tl.dot(Xh[p, all_but_pq], Xh[q, all_but_pq]) - tl.dot(
                Xh[all_but_pq, p], Xh[all_but_pq, q]
            )
            Gh = (
                tl.norm(Xh[p, all_but_pq]) ** 2
                + tl.norm(Xh[q, all_but_pq]) ** 2
                + tl.norm(Xh[all_but_pq, p]) ** 2
                + tl.norm(Xh[all_but_pq, q]) ** 2
            )
            xih = Xh[p, q] - Xh[q, p]

            # Build shearing matrix out of these quantities
            yk = tl.arctanh((Kh - xih * dh) / (2 * (dh**2 + xih**2) + Gh))

            # Inverse of Sk on left side
            pvec = tl.copy(X[p, :, :])
            X = tl.index_update(
                X,
                tl.index[p, :, :],
                X[p, :, :] * tl.cosh(yk) - X[q, :, :] * tl.sinh(yk),
            )
            X = tl.index_update(
                X, tl.index[q, :, :], -pvec * tl.sinh(yk) + X[q, :, :] * tl.cosh(yk)
            )

            # Sk on right side
            pvec = tl.copy(X[:, p, :])
            X = tl.index_update(
                X,
                tl.index[:, p, :],
                X[:, p, :] * tl.cosh(yk) + X[:, q, :] * tl.sinh(yk),
            )
            X = tl.index_update(
                X, tl.index[:, q, :], pvec * tl.sinh(yk) + X[:, q, :] * tl.cosh(yk)
            )

            # Update Q_total
            pvec = tl.copy(Q_total[:, p])
            Q_total = tl.index_update(
                Q_total,
                tl.index[:, p],
                Q_total[:, p] * tl.cosh(yk) + Q_total[:, q] * tl.sinh(yk),
            )
            Q_total = tl.index_update(
                Q_total,
                tl.index[:, q],
                pvec * tl.sinh(yk) + Q_total[:, q] * tl.cosh(yk),
            )

            # Defines array of off-diagonal element differences
            xi_ = -X[q, p, :] - X[p, q, :]

            # More quantities computed
            Esum = 2 * tl.dot(xi_, d_)
            Dsum = tl.dot(d_, d_) - tl.dot(xi_, xi_)
            qt = Esum / Dsum

            th1 = tl.arctan(qt)
            angle_selection = tl.cos(th1) * Dsum + tl.sin(th1) * Esum

            # Defines 1 of 2 possible angles
            if angle_selection > 0.0:
                theta_k = th1 / 4
            elif angle_selection < 0.0:
                theta_k = (th1 + tl.pi) / 4
            else:
                raise RuntimeError("Jointdiag: No solution found.")

            # Given's rotation, this will minimize norm of off-diagonal elements only
            pvec = tl.copy(X[p, :, :])
            X = tl.index_update(
                X,
                tl.index[p, :, :],
                X[p, :, :] * tl.cos(theta_k) - X[q, :, :] * tl.sin(theta_k),
            )
            X = tl.index_update(
                X,
                tl.index[q, :, :],
                pvec * tl.sin(theta_k) + X[q, :, :] * tl.cos(theta_k),
            )

            pvec = tl.copy(X[:, p, :])
            X = tl.index_update(
                X,
                tl.index[:, p, :],
                X[:, p, :] * tl.cos(theta_k) - X[:, q, :] * tl.sin(theta_k),
            )
            X = tl.index_update(
                X,
                tl.index[:, q, :],
                pvec * tl.sin(theta_k) + X[:, q, :] * tl.cos(theta_k),
            )

            # Update Q_total
            pvec = tl.copy(Q_total[:, p])
            Q_total = tl.index_update(
                Q_total,
                tl.index[:, p],
                Q_total[:, p] * tl.cos(theta_k) - Q_total[:, q] * tl.sin(theta_k),
            )
            Q_total = tl.index_update(
                Q_total,
                tl.index[:, q],
                pvec * tl.sin(theta_k) + Q_total[:, q] * tl.cos(theta_k),
            )

        # Error computation, check if loop needed...
        old_e = e
        e = tl.norm(X) ** 2.0 - tl.norm(tl.diagonal(X, axis1=1, axis2=2)) ** 2.0

        if verbose:
            print(f"Sweep # {k + 1}: e = {e:.3e}")

        # TODO: Strangely the error increases on the first iteration
        if old_e - e < threshold and k > 2:
            break

    return X, Q_total
