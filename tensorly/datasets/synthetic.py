import numpy as np
from .. import backend as T


def gen_image(region='swiss', image_height=20, image_width=20,
              n_channels=None, weight_value=1):
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
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        weight[cy - radius:cy + radius, cx - radius:cx + radius][index] = 1

    if n_channels is not None and weight.ndim == 2:
        weight = np.concatenate([weight[..., None]] * n_channels, axis=-1)

    return T.tensor(weight)
