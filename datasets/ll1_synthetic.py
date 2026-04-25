"""
Synthetic data generation for LL1 tensor decomposition.

Provides utilities for generating third-order tensors that follow an exact
LL1 model with factors ``A`` (I, L*R), ``B`` (J, L*R), ``C`` (K, R),
optionally with Stokes constraints on the columns of ``C`` and
non-negativity on ``A`` and ``B``.
"""

import numpy as np
from .. import backend as T
from ..ll1_tensor import LL1Tensor, ll1_to_tensor


def gen_ll1(
    shape, rank, column_rank, noise_level=0.0, stokes=False, random_state=None
):
    r"""Generate a synthetic third-order tensor with an exact LL1 structure.

    The tensor is constructed as::

        X[:, :, k] = sum_{r=1}^{R}  C[k, r] * A_r @ B_r^T  +  noise

    Parameters
    ----------
    shape : tuple ``(I, J, K)``
        Shape of the generated tensor.  When ``stokes=True``, ``K``
        must be 4.
    rank : int
        Number of LL1 terms ``R``.
    column_rank : int
        Column rank ``L`` of each matrix factor block.
    noise_level : float, optional
        Relative noise level (standard deviation as a fraction of the
        noiseless tensor norm).  ``0.0`` means no noise.
    stokes : bool, optional
        If ``True``, each column of ``C`` is a valid Stokes vector
        and ``A``, ``B`` are non-negative.  Requires ``K == 4``.
    random_state : {None, int, np.random.RandomState}

    Returns
    -------
    tensor : ndarray of shape ``(I, J, K)``
        The (possibly noisy) tensor.
    ll1_ground_truth : LL1Tensor
        The ground-truth (noiseless) decomposition ``(A, B, C)``.

    Raises
    ------
    ValueError
        If ``stokes=True`` and ``K != 4``.
    """
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    I, J, K = shape
    R = rank
    L = column_rank

    if stokes and K != 4:
        raise ValueError(
            f"Stokes-constrained LL1 requires K == 4, got K = {K}."
        )

    if stokes:
        # Non-negative A and B
        A_np = np.abs(rng.random_sample((I, L * R)))
        B_np = np.abs(rng.random_sample((J, L * R)))

        # Valid Stokes columns: s0 = 1, s1..s3 each in [0, 0.5]
        # so that s0^2 = 1 >= s1^2 + s2^2 + s3^2 <= 0.75
        C_np = np.zeros((4, R))
        for r in range(R):
            s1, s2, s3 = rng.random_sample(3) * 0.5
            C_np[:, r] = [1.0, s1, s2, s3]
    else:
        A_np = rng.random_sample((I, L * R))
        B_np = rng.random_sample((J, L * R))
        C_np = rng.random_sample((K, R))

    A = T.tensor(A_np)
    B = T.tensor(B_np)
    C = T.tensor(C_np)

    ll1_ground_truth = LL1Tensor((A, B, C))
    tensor_clean = ll1_to_tensor(ll1_ground_truth)

    if noise_level > 0.0:
        noise = T.tensor(rng.random_sample(shape)) * 2.0 - 1.0
        noise = noise / T.norm(noise, 2) * T.norm(tensor_clean, 2) * noise_level
        tensor = tensor_clean + noise
    else:
        tensor = tensor_clean

    return tensor, ll1_ground_truth
