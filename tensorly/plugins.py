import tensorly as tl

# Author: Jean Kossaifi


PREVIOUS_EINSUM = None
OPT_EINSUM_PATH_CACHE = dict()
CUQUANTUM_PATH_CACHE = dict()


def use_default_einsum():
    """Revert to the original einsum for the current backend"""
    global PREVIOUS_EINSUM

    tl.backend.BackendManager.register_backend_method("einsum", PREVIOUS_EINSUM)
    PREVIOUS_EINSUM = None


def use_opt_einsum(optimize="auto-hq"):
    """Plugin to use opt-einsum [1]_ to precompute (and cache) a better contraction path

    Examples
    --------
    >>> import tensorly as tl

    Use your favourite backend, here PyTorch:
    >>> tl.set_backend('pytorch')

    Use the convenient backend system to automatically dispatch all tenalg operations to einsum
    >>> from tensorly import tenalg
    >>> tenalg.set_backend('einsum')

    Now you can transparently cache the optimal contraction path, transparently:
    >>> from tensorly import plugins
    >>> plugins.use_opt_einsum()

    That's it! You can revert to the original einsum just as easily:
    >>> plugings.use_default_einsum()

    Revert to the original tensor algebra backend:
    >>> tenalg.set_backend('core')

    References
    ----------
    .. [1] Daniel G. A. Smith and Johnnie Gray, opt_einsum,
           A Python package for optimizing contraction order for einsum-like expressions.
           Journal of Open Source Software, 2018, 3(26), 753
    """
    global PREVIOUS_EINSUM

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
        key = equation + "," + ",".join(f"{s}" for s in shapes)
        try:
            expression = OPT_EINSUM_PATH_CACHE[key]
        except KeyError:
            expression = oe.contract_expression(equation, *shapes, optimize=optimize)
            OPT_EINSUM_PATH_CACHE[key] = expression

        return expression(*args)

    if PREVIOUS_EINSUM is None:
        PREVIOUS_EINSUM = tl.backend.current_backend().einsum

    tl.backend.BackendManager.register_backend_method("einsum", cached_einsum)
