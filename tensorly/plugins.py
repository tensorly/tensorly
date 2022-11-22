import tensorly as tl

# Author: Jean Kossaifi


OPT_EINSUM_PATH_CACHE = dict()


def use_opt_einsum(optimize="auto-hq"):
    """Plugin to use opt-einsum to precompute (and cache) a better contraction path"""
    try:
        import opt_einsum as oe
    except ImportError as error:
        message = (
            "Impossible to import opt-einsum.\n"
            "First install it:\n"
            "conda install opt_einsum -c conda-forge\n"
            " or pip install opt_einsum"
        )
        raise ImportError(message) from error

    def cached_einsum(equation, *args):
        shapes = [tl.shape(arg) for arg in args]
        key = ",".join(f"{s}" for s in shapes)
        try:
            expression = OPT_EINSUM_PATH_CACHE[key]
        except KeyError:
            expression = oe.contract_expression(equation, *shapes, optimize="optimal")
            OPT_EINSUM_PATH_CACHE[key] = expression

        return expression(*args)

    tl.backend.BackendManager.register_backend_method("einsum", cached_einsum)
