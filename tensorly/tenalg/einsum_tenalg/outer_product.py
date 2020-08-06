from ... import backend as tl

# Author: Jean Kossaifi

# License: BSD 3 clause

def outer(vectors):
    """Returns the outer product of vectors

    Parameters
    ----------
    vectors : 1-D tensor list
        list of vectors

    Returns
    -------
    tensor of order len(vectors) with tensor.shape[i] == len(vectors[i])
    """
    start = ord('a')
    symbols = [chr(start + i) for i in range(len(vectors))]
    equation = ','.join(symbols) + '->' + ''.join(symbols)
    return tl.einsum(equation, *vectors)