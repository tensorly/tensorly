EINSUM_PATH_CACHE = dict()


def einsum_path_cached(fun):
    def wrapped(key, *args, **kwargs):
        name = fun.__name__
        try:
            cache = EINSUM_PATH_CACHE[name]
        except KeyError:
            EINSUM_PATH_CACHE[name] = dict()
            cache = EINSUM_PATH_CACHE[name]
        try:
            equation = cache[key]
        except KeyError:
            equation = fun(*args, **kwargs)
            cache[key] = equation

        return equation

    return wrapped
