import tensorly as tl


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
    batch = "a"
    start = ord(batch) + 1
    n_features = tl.ndim(tensor) - 1
    feature_syms = [
        "".join(chr(start + i * n_features + j) for j in range(n_features))
        for i in range(order)
    ]
    eq = ",".join(batch + sym for sym in feature_syms) + "->" + "".join(feature_syms)

    return tl.einsum(eq, *([tensor] * order)) / tl.shape(tensor)[0]
