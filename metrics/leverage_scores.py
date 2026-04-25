import tensorly as tl


def leverage_score_dist(matrix):
    """Compute leverage score distribution over the rows of `matrix`

    See the reference [1]_ for an overview on the usage of leverage scores in randomized
    numerical linear algebra.

    When the leverage score distribution is given as an input to rng.choice, where
    rng = tl.check_random_state(random_state), rng.choice can complain that the
    distribution doesn't sum to 1 if single precision is used for the probability
    vector. For that reason, leverage_score_dist always returns the distribution in
    double precision format.

    Parameters
    ----------
    matrix : tl.tensor
        As the parameter name implies, `matrix` needs to be a tl.tensor with two modes.

    Returns
    -------
    lev_score_dist : tl.tensor
        The leverage scores in a vector in tl.tensor format. The dtype will be
        tl.float64, even if the input `matrix` is lower precision.

    References
    ----------
    .. [1] P. Drineas, M. W. Mahoney, "RandNLA: randomized numerical linear algebra",
           Commun. ACM 59(6), pp. 80-90, 2016. DOI: 10.1145/2842602
    """

    U, S, _ = tl.svd(matrix, full_matrices=False)
    mat_dtype = tl.context(matrix)["dtype"]
    rank_cutoff = tl.max(S) * max(matrix.shape) * tl.eps(mat_dtype)
    num_rank = (
        int(tl.max(tl.where(S > rank_cutoff)[0])) + 1
    )  # int(...) needed for mxnet
    lev_score_dist = tl.sum(U[:, :num_rank] ** 2, axis=1) / tl.tensor(
        num_rank, dtype=mat_dtype
    )

    if tl.context(lev_score_dist)["dtype"] != tl.float64:
        lev_score_dist = tl.tensor(lev_score_dist, dtype=tl.float64)
        lev_score_dist /= tl.sum(lev_score_dist)

    return lev_score_dist
