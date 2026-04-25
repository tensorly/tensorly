"""
Visualization utilities for tensor decompositions.

This module provides plotting functions for inspecting the output of tensor
decomposition methods, with a focus on LL1 decompositions.
"""

import numpy as np


def plot_ll1_terms(ll1_tensor, cmap="viridis", suptitle="LL1 Components"):
    r"""Visualise the R terms of an LL1 decomposition.

    For each component ``r`` three subplots are shown left to right:

    * **Left**   – the matrix block ``A_r``  ``(I x L)`` as a heat-map.
    * **Centre** – the matrix block ``B_r``  ``(J x L)`` as a heat-map.
    * **Right**  – the vector ``c_r = C[:, r]``  ``(K,)`` as a bar chart.

    Parameters
    ----------
    ll1_tensor : LL1Tensor or tuple ``(A, B, C)``
        The LL1 decomposition to visualise.
    cmap : str, optional
        Matplotlib colour-map for the heat-maps.
    suptitle : str, optional
        Title placed above the whole figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
        Array of shape ``(R, 3)``.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )

    from . import backend as T
    from .ll1_tensor import _validate_ll1_tensor

    # Accept both LL1Tensor instances and raw (A, B, C) tuples
    if hasattr(ll1_tensor, "A"):
        A, B, C = ll1_tensor.A, ll1_tensor.B, ll1_tensor.C
        R = ll1_tensor.rank
        L = ll1_tensor.column_rank
    else:
        _, R, L = _validate_ll1_tensor(ll1_tensor)
        A, B, C = ll1_tensor

    fig, axes = plt.subplots(R, 3, figsize=(13, 3 * R), squeeze=False)
    fig.suptitle(suptitle, fontsize=14, y=1.02)

    A_np = T.to_numpy(A)
    B_np = T.to_numpy(B)
    C_np = T.to_numpy(C)

    for r in range(R):
        A_r = A_np[:, r * L : (r + 1) * L]
        B_r = B_np[:, r * L : (r + 1) * L]
        c_r = C_np[:, r]

        # A_r heat-map
        ax = axes[r, 0]
        im = ax.imshow(A_r, aspect="auto", cmap=cmap)
        ax.set_title(rf"$A_{{r={r + 1}}}$  ({A_r.shape[0]}$\times${A_r.shape[1]})")
        ax.set_xlabel("l")
        ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, shrink=0.8)

        # B_r heat-map
        ax = axes[r, 1]
        im = ax.imshow(B_r, aspect="auto", cmap=cmap)
        ax.set_title(rf"$B_{{r={r + 1}}}$  ({B_r.shape[0]}$\times${B_r.shape[1]})")
        ax.set_xlabel("l")
        ax.set_ylabel("j")
        fig.colorbar(im, ax=ax, shrink=0.8)

        # c_r bar chart
        ax = axes[r, 2]
        ax.bar(range(len(c_r)), c_r, color="steelblue")
        ax.set_title(rf"$c_{{r={r + 1}}}$")
        ax.set_xlabel("k")
        ax.set_ylabel("value")

    plt.tight_layout()
    return fig, axes
