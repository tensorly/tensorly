from functools import reduce

def _prod(array_like):
    """Calculate the product of all the elements in the input iterable. The default start value for the product is 1.

    When the iterable is empty, return the start value.

    Parameters
    ----------
    array_like : list-like object

    Returns
    -------
    equivalent of math.prod(array_like)
    """
    
    return reduce(lambda x, y: x*y, array_like, 1)


try:
    from math import prod
except ImportError:
    prod = _prod

