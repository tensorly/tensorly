from .. import random
import warnings


def check_random_state(seed):
    """Function moved: Use tensorly.random.check_random_state instead.
    
        Here only for backward compatibility only.
    """
    warnings.warn('Function moved: check_random_state has been moved to '
                  'the module tensorly.random.')
    return random.check_random_state(seed)
