import tensorly as tl
from .caching import einsum_path_cached


@einsum_path_cached
def higher_order_moment_path(tensor, order):
    batch = ord("a")
    start = batch + 1
    tensor_sym = [f"{chr(start+i)}" for i, _ in enumerate(tensor.shape)]
    out_sym = tensor_sym[1:] * order
    eq = "".join(tensor_sym) + "->" + "".join(out_sym)
    return eq


def higher_order_moment(tensor, order):
    """Computes the Higher-Order Momemt

    Parameters
    ----------
    tensor : 2D-tensor -- or ND-tensor
        matrix of size (n_samples, n_features)
        or tensor of size(n_samples, D1, ..., DN)

    order : int
        order of the higher-order moment to compute

    Returns
    -------
    tensor : moment
        if tensor is a matrix of size (n_samples, n_features),
        tensor of size (n_features, )*order
    """
    key = f"{tl.shape(tensor)},{order}"
    eq = higher_order_moment_path(key, tensor, order)
    return tl.einsum(eq, tensor)
