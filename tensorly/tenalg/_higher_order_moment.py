from ._khatri_rao import khatri_rao


def higher_order_moment(matrix, order=3):
    """Higher order moment of a matrix of observations

        Computes the `order`-order moment of `matrix``
        Each row of `matrix` represents a samples
        (i.e. an observation)

    Parameters
    ----------
    matrix : 2D-array
        array of shape (n_samples, n_features)
        i.e. each row is a sample
    order : int, optional
        order of the moment to compute
    """
    matrix = matrix - matrix.mean(axis=0)
    n_features = matrix.shape[-1]
    t = khatri_rao([matrix.T] * order).mean(axis=1)
    return t.reshape([n_features] * order)
