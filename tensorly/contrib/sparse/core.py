import functools

from .backend import sparse_context

def wrap(func):
    @functools.wraps(func, assigned=('__name__', '__qualname__',
                                     '__doc__', '__annotations__'))
    def inner(*args, **kwargs):
        with sparse_context():
            return func(*args, **kwargs)

    return inner
