import numpy as np


def check_random_state(seed):
    """Returns a valid RandomState

    Parameters
    ----------
    seed : None or instance of int or np.random.RandomState(), default is None

    Returns
    -------
    Valid instance np.random.RandomState

    Notes
    -----
    Inspired by the scikit-learn eponymous function
    """
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)

    elif isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('Seed should be None, int or np.random.RandomState')
