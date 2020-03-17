from .. import backend as T

# Author: Jean Kossaifi

# License: BSD 3 clause

def outer(vectors, weights=None):
    """Returns the outer product of vectors

    Parameters
    ----------
    vectors : 1-D tensor list
        list of vectors

    Returns
    -------
    tensor of order len(vectors) with tensor.shape[i] == len(vectors[i])
    """
    order = len(vectors)
    shapes = [[-1 if i == j else 1 for i in range(order)] for j in range(order)]
    for i, v in enumerate(vectors):
        if not i:
            if weights is None:
                res = T.reshape(v, shapes[0])
            else:
                res = T.reshape(v*weights, shapes[0])
        else:
            res = res*T.reshape(v, shapes[i])
    return res