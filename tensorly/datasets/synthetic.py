import numpy as np
from ..utils import check_random_state


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
    weight = np.zeros((image_height, image_width), dtype=np.float)

    if region is "swiss":
        slim_width = (image_width // 2) - (image_width // 10 + 1)
        large_width = (image_width // 2) - (image_width // 3 + 1)
        slim_height = (image_height // 2) - (image_height // 10 + 1)
        large_height = (image_height // 2) - (image_height // 3 + 1)
        weight[large_height:-large_height, slim_width:-slim_width] = weight_value
        weight[slim_height:-slim_height, large_width:-large_width] = weight_value

    elif region is "rectangle":
        large_height = (image_height // 2) - (image_height // 4)
        large_width = (image_width // 2) - (image_width // 4)
        weight[large_height:-large_height, large_width:-large_width] = weight_value

    elif region is "circle":
        radius = int(image_width // 3)
        cy = int(image_width / 2)
        cx = int(image_height / 2)
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        weight[cy-radius:cy+radius, cx-radius:cx+radius][index] = 1

    if n_channels is not None and weight.ndim == 2:
        weight = np.concatenate([weight[..., None]]*n_channels, axis=-1)

    return weight



def low_rank_matrix(size, rank, random_state=None):
    """Generates a random low-rank matrix

    Parameters
    ----------
    size : (int, int)
        (n_rows, n_columns) = size of the matrix to be created
    rank : int
        rank of the matrix to be generated
    random_state : `np.random.RandomState`

    Returns
    -------
    low_rank_matrix : ndarray of shape size

        matrix with the specified rank
    """
    rns = check_random_state(random_state)
    right_factor = rns.rand(size[0], rank)
    left_factor = rns.rand(rank, size[1])
    return np.dot(right_factor, left_factor)
