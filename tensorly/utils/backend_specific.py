from functools import wraps
from .. import get_backend, available_backend_names
from inspect import getfullargspec


def register_backend_specific(backend, backend_func):
    """A decorator to create a backend specific function.

    Parameters
    ----------
    backend : str
        String that specifies which backend implementation to overload. Must be an entry
        in the ``tensorly.available_backend_names``-list.
    backend_func : callable
        Function implementation for the given backend. The function signature must match
        the signature of the backend agnostic implementation exactly (including which
        arguments are keyword-/positional-only and the default arguments) to ensure that
        the documentation is correct.

    Returns
    -------
    func
        Decorator that creates a backend specific function

    Examples
    --------
    >>> from tensorly.backend import backend_context
    >>> from tensorly.utils.backend_specific import register_backend_specific
    >>> def _numpy_func(x):
    ...    return 2
    >>> @register_backend_specific("numpy", _numpy_func)
    ... def func(x):
    ...     return x
    >>> with backend_context("numpy"):
    ...     print(func(5))
    2
    >>> with backend_context("pytorch"):
    ...     print(func(5))
    5
    >>> with backend_context("paddle"):
    ...     print(func(5))
    5
    """
    if backend not in available_backend_names:
        raise ValueError(
            f"Unknown backend name '{backend}', "
            + f"known backends are {available_backend_names}"
        )

    def wrapper(f):
        if getfullargspec(f) != getfullargspec(backend_func):
            raise TypeError(
                "Signature of backend spesific implementation does not match the "
                + "signature of the backend agnostic implementation. All arguments and "
                + "default values must match, otherwise the documentation will be"
                + "incorrect."
            )

        @wraps(f)
        def wrapped(*args, **kwargs):
            if get_backend() == backend:
                return backend_func(*args, **kwargs)
            return f(*args, **kwargs)

        return wrapped

    return wrapper
