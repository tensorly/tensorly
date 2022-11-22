from collections import defaultdict


EINSUM_PATH_CACHE = defaultdict(dict)


def einsum_path_cached(fun):
    def wrapped(key, *args, **kwargs):
        name = fun.__name__
        cache = EINSUM_PATH_CACHE[name]
        try:
            equation = cache[key]
        except KeyError:
            equation = fun(*args, **kwargs)
            cache[key] = equation

        return equation

    return wrapped
