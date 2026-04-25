import numpy as np
from .. import backend as T
import tensorly as tl


def gen_image(
    region="swiss", image_height=20, image_width=20, n_channels=None, weight_value=1
):
    """Generates an image for regression testing

    Parameters
    ----------
    region : {'swiss', 'rectangle'}
    image_height : int, optional
    image_width : int, optional
    weight_value : float, optional
    n_channels : int or None, optional
        if not None, the resulting image will have a third dimension

    Returns
    -------
    ndarray
        array of shape ``(image_height, image_width)``
        or ``(image_height, image_width, n_channels)``
        array for which all values are zero except the region specified
    """
    weight = np.zeros((image_height, image_width), dtype=float)

    if region == "swiss":
        slim_width = (image_width // 2) - (image_width // 10 + 1)
        large_width = (image_width // 2) - (image_width // 3 + 1)
        slim_height = (image_height // 2) - (image_height // 10 + 1)
        large_height = (image_height // 2) - (image_height // 3 + 1)
        weight[large_height:-large_height, slim_width:-slim_width] = weight_value
        weight[slim_height:-slim_height, large_width:-large_width] = weight_value

    elif region == "rectangle":
        large_height = (image_height // 2) - (image_height // 4)
        large_width = (image_width // 2) - (image_width // 4)
        weight[large_height:-large_height, large_width:-large_width] = weight_value

    elif region == "circle":
        radius = image_width // 3
        cy = image_width // 2
        cx = image_height // 2
        y, x = np.ogrid[-radius:radius, -radius:radius]
        index = x**2 + y**2 <= radius**2
        weight[cy - radius : cy + radius, cx - radius : cx + radius][index] = 1

    if n_channels is not None and weight.ndim == 2:
        weight = np.concatenate([weight[..., None]] * n_channels, axis=-1)

    return T.tensor(weight)


def gen_ll1(
    shape,
    rank,
    stokes_constrained=False,
    non_negative_matrices=False,
    noise_level=0.0,
    random_state=None,
):
    r"""Generate a synthetic tensor following the LL1 model.

    Constructs a rank-*R* LL1 tensor of shape ``(I, J, K)`` as

    .. math::

        \mathcal{T} = \sum_{r=1}^{R} \mathbf{A}_r \otimes \mathbf{c}_r,

    where each :math:`\mathbf{A}_r` is an ``(I, J)`` activation matrix and
    :math:`\mathbf{c}_r` is a *K*-vector.  When ``stokes_constrained=True``
    the vectors :math:`\mathbf{c}_r` are valid Stokes vectors (requires
    ``K == 4``).

    Parameters
    ----------
    shape : tuple of int ``(I, J, K)``
        Desired tensor shape.
    rank : int
        Number of LL1 terms *R*.
    stokes_constrained : bool, optional
        If ``True``, each mode-*K* vector is a valid Stokes vector
        ``[S0, S1, S2, S3]`` with ``S0 > 0`` and
        ``S0 >= ||(S1, S2, S3)||``.  Requires ``K == 4``.  Default ``False``.
    non_negative_matrices : bool, optional
        If ``True``, the activation matrices :math:`\mathbf{A}_r` are
        drawn from a uniform distribution on ``[0, 1]``.  If ``False``
        (default), entries are drawn from ``[-1, 1]``.
    noise_level : float, optional
        If ``> 0``, additive Gaussian noise is scaled so that
        ``||noise||_F = noise_level * ||T_clean||_F``.  Default ``0``.
    random_state : None, int, or ``RandomState``, optional

    Returns
    -------
    tensor : ndarray of shape ``(I, J, K)``
        Generated (possibly noisy) tensor.
    matrices : list of ndarray
        *R* activation matrices, each of shape ``(I, J)``.
    vectors : ndarray of shape ``(K, R)``
        Matrix of mode-*K* (Stokes) vectors.

    Examples
    --------
    >>> tensor, matrices, vectors = gen_ll1((10, 8, 4), rank=3,
    ...                                     stokes_constrained=True,
    ...                                     non_negative_matrices=True,
    ...                                     random_state=0)
    >>> tensor.shape
    (10, 8, 4)
    >>> len(matrices)
    3
    """
    rng = tl.check_random_state(random_state)

    if len(shape) != 3:
        raise ValueError(f"shape must be a 3-tuple (I, J, K), got {shape}.")

    I, J, K = shape

    if stokes_constrained and K != 4:
        raise ValueError(
            f"Stokes-constrained generation requires K=4, got K={K}."
        )

    # --- Activation matrices -----------------------------------------------
    if non_negative_matrices:
        matrices_np = [rng.random_sample((I, J)) for _ in range(rank)]
    else:
        matrices_np = [rng.random_sample((I, J)) * 2.0 - 1.0 for _ in range(rank)]

    # --- Mode-K vectors -------------------------------------------------------
    vectors_np = np.zeros((K, rank))
    if stokes_constrained:
        for r in range(rank):
            # Intensity S0 drawn from Uniform[0.5, 1.5]
            S0 = rng.random_sample() + 0.5
            # Degree of polarization in [0, 1)
            dop = rng.random_sample()
            # Random unit polarization direction in R^3
            p = rng.standard_normal(3)
            p = p / (np.linalg.norm(p) + 1e-14)
            vectors_np[:, r] = [
                S0,
                dop * S0 * p[0],
                dop * S0 * p[1],
                dop * S0 * p[2],
            ]
    else:
        vectors_np = rng.random_sample((K, rank)) * 2.0 - 1.0

    # --- Assemble tensor T = sum_r A_r ⊗ c_r ----------------------------------
    tensor_np = np.zeros((I, J, K))
    for r in range(rank):
        # A_r[:, :, None] * c_r[None, None, :] → (I, J, K)
        tensor_np += (
            matrices_np[r][:, :, np.newaxis]
            * vectors_np[:, r][np.newaxis, np.newaxis, :]
        )

    # --- Optional noise -------------------------------------------------------
    if noise_level > 0.0:
        noise = rng.standard_normal((I, J, K))
        tensor_norm = np.linalg.norm(tensor_np)
        noise_norm = np.linalg.norm(noise)
        if noise_norm > 0:
            tensor_np = tensor_np + noise_level * tensor_norm * noise / noise_norm

    matrices = [tl.tensor(m) for m in matrices_np]
    vectors = tl.tensor(vectors_np)
    tensor = tl.tensor(tensor_np)

    return tensor, matrices, vectors
