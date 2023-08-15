import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..base import matricize
from ..tr_tensor import validate_tr_rank, TRTensor
from ..tenalg.svd import svd_interface


def tensor_ring(input_tensor, rank, mode=0, svd="truncated_svd", verbose=False):
    """Tensor Ring decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors) [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : Union[int, List[int]]
            maximum allowable TR rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    mode : int, default is 0
            index of the first factor to compute
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TR factors
              order-3 tensors of the TR decomposition

    References
    ----------
    .. [1] Qibin Zhao et al. "Tensor Ring Decomposition" arXiv preprint arXiv:1606.05535, (2016).
    """
    rank = validate_tr_rank(tl.shape(input_tensor), rank=rank)
    n_dim = len(input_tensor.shape)

    # Change order
    if mode:
        order = tuple(range(mode, n_dim)) + tuple(range(mode))
        input_tensor = tl.transpose(input_tensor, order)
        rank = rank[mode:] + rank[:mode]

    tensor_size = input_tensor.shape

    factors = [None] * n_dim

    # Getting the first factor
    unfolding = tl.reshape(input_tensor, (tensor_size[0], -1))

    n_row, n_column = unfolding.shape
    if rank[0] * rank[1] > min(n_row, n_column):
        raise ValueError(
            f"rank[{mode}] * rank[{mode + 1}] = {rank[0] * rank[1]} is larger than "
            f"first matricization dimension {n_row}×{n_column}.\n"
            "Failed to compute first factor with specified rank. "
            "Reduce specified ranks or change first matricization `mode`."
        )

    # SVD of unfolding matrix
    U, S, V = svd_interface(unfolding, n_eigenvecs=rank[0] * rank[1], method=svd)

    # Get first TR factor
    factor = tl.reshape(U, (tensor_size[0], rank[0], rank[1]))
    factors[0] = tl.transpose(factor, (1, 0, 2))
    if verbose is True:
        print("TR factor " + str(mode) + " computed with shape " + str(factor.shape))

    # Get new unfolding matrix for the remaining factors
    unfolding = tl.reshape(S, (-1, 1)) * V
    unfolding = tl.reshape(unfolding, (rank[0], rank[1], -1))
    unfolding = tl.transpose(unfolding, (1, 2, 0))

    # Getting the TR factors up to n_dim - 1
    for k in range(1, n_dim - 1):
        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        n_row, n_column = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        U, S, V = svd_interface(unfolding, n_eigenvecs=current_rank, method=svd)
        rank[k + 1] = current_rank

        # Get kth TR factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))

        if verbose is True:
            print(
                "TR factor "
                + str((mode + k) % n_dim)
                + " computed with shape "
                + str(factors[k].shape)
            )

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    prev_rank = unfolding.shape[0]
    factors[-1] = tl.reshape(unfolding, (prev_rank, -1, rank[0]))

    if verbose is True:
        print(
            "TR factor "
            + str((mode - 1) % n_dim)
            + " computed with shape "
            + str(factors[-1].shape)
        )

    # Reorder factors to match input
    if mode:
        factors = factors[-mode:] + factors[:-mode]

    return TRTensor(factors)


def tensor_ring_als(
    tensor,
    rank,
    ls_solve="lstsq",
    n_iter_max=100,
    tol=1e-6,
    random_state=None,
    verbose=False,
    callback=None,
):
    """Tensor ring decomposition via alternating least squares (ALS)

    Computes a rank-`rank` tensor ring decomposition of `tensor` using ALS. The
    implementation roughly follows Algorithm 2 in [1]_.

    Parameters
    ----------
    tensor : ndarray
    rank : Union[int, List[int]]
        The rank of the decomposition. If `rank` is an int, then all ranks will be the
        same and equal to `rank`. If `rank` is a list, then the i-th core will be of
        size rank[i]-by-shape[i]-by-rank[i+1], where shape[i] is the dimension of the
        i-th mode of `tensor`.
    ls_solve : {"lstsq", "normal_eq"}, default is "lstsq"
        When equal to "lstsq", the least squares problems are solved as overdetermined
        systems of equations using tl.lstsq. When equal to "normal_eq", a normal
        equation formulation is used instead together with tl.solve. This latter
        approach is expected to yield poorer accuracy due to numerical issuse with the
        normal equations.
    n_iter_max : int, default is 100
        Maximum number of ALS iterations.
    tol : float, default 1e-6
        The algorithm is terminated when the change in the relative reconstruction error
        is less than `tol`.
    random_state : {None, int, np.random.RandomState}
        Used to set the random seed in the algorithm.
    verbose : bool, default False
        If True, the algorithm will make some additional print outs.
    callback : {None, Callable[[tl.tr_tensor.TRTensor, float], {None, bool}]}, default None
        A callback function which can be used for, e.g., logging the per-iteration
        error. Such a function takes two inputs: A tensor ring decomposition and a float
        which indicates the relative reconstruction error between `tensor` and the
        inputted TRTensor. It can return a bool which can be used to control when the
        ALS algorithm terminates.

    Returns
    -------
    tr_decomp : tl.tr_tensor.TRTensor
        The tensor ring decomposition computed by the algorithm.

    References
    ----------
    .. [1] Q. Zhao, G. Zhou, S. Xie, L. Zhang, A. Cichocki, "Tensor Ring Decomposition",
           arXiv:1606.05535, 2016.
    """

    shape = tl.shape(tensor)
    rank = validate_tr_rank(shape, rank=rank)
    n_dim = len(shape)
    rng = tl.check_random_state(random_state)
    if tol > 0 or callback:
        tensor_norm = tl.norm(tensor)
    valid_ls_solve = {"lstsq", "normal_eq"}
    if ls_solve not in valid_ls_solve:
        raise ValueError(
            f"Invalid value provided for ls_solve. It must be in {valid_ls_solve}."
        )

    # Randomly initialize decomposition cores
    tr_decomp = tl.random.random_tr(shape, rank, random_state=rng, **tl.context(tensor))

    # Run callback function if provided
    if callback:
        rel_error = tl.norm(tl.tr_to_tensor(tr_decomp) - tensor) / tensor_norm
        callback(tr_decomp, rel_error)

    # Main loop
    rec_errors = []
    for iter in range(n_iter_max):
        for dim in range(n_dim):
            # Compute appropriate transposed unfolding of tensor
            tensor_unf = matricize(tensor, [n for n in range(n_dim) if n != dim], [dim])

            # Compute design matrix
            subchain_tensor = tr_decomp[(dim + 1) % n_dim]
            for j in range(2, n_dim):
                subchain_tensor = tl.tensordot(
                    subchain_tensor, tr_decomp[(dim + j) % n_dim], axes=1
                )
            tr_idx = (
                [i + n_dim - dim for i in range(dim)]
                + [i + 1 for i in range(n_dim - dim - 1)]
                + [n_dim, 0]
            )
            subchain_tensor = tl.transpose(subchain_tensor, tr_idx)
            design_mat = tl.reshape(subchain_tensor, (-1, rank[dim] * rank[dim + 1]))

            if ls_solve == "lstsq":
                # Solve least squares problem directly
                sol, *_ = tl.lstsq(design_mat, tensor_unf)

            elif ls_solve == "normal_eq":
                # Solve least squares problem via normal equations
                design_mat_tr = tl.transpose(design_mat)
                gram_mat = tl.matmul(design_mat_tr, design_mat)
                rhs_mat = tl.matmul(design_mat_tr, tensor_unf)
                sol = tl.solve(gram_mat, rhs_mat)

            # Update core
            tr_decomp[dim] = tl.transpose(
                tl.reshape(sol, (rank[dim], rank[dim + 1], shape[dim])),
                [0, 2, 1],
            )

        # Compute relative error if necessary
        if tol > 0 or callback:
            error = tl.norm(tl.matmul(design_mat, sol) - tensor_unf)
            rel_error = error / tensor_norm
            rec_errors.append(rel_error)
            if iter >= 1:
                rel_error_decrease = rec_errors[-2] - rec_errors[-1]
            if verbose:
                if iter >= 1:
                    print(
                        f"Iteration {iter+1} finished. Reconstruction error: {rel_error}, decrease = {rel_error_decrease}, unnormalized = {error}"
                    )
                else:
                    print(
                        f"Iteration {iter+1} finished. Reconstruction error: {rel_error}, unnormalized = {error}"
                    )
        elif verbose:
            print(f"Iteration {iter+1} finished.")

        # Run callback function if provided
        if callback:
            callback_retVal = callback(tr_decomp, rel_error)
            if callback_retVal:
                if verbose:
                    print("Received True from callback function. Exiting.")
                break

        # Check convergence
        if tol > 0 and iter >= 1:
            if rel_error_decrease < tol:
                if verbose:
                    print(f"tensor_ring_als converged after {iter} iterations.")
                break

    return tr_decomp


class TensorRing(DecompositionMixin):
    """Tensor Ring decomposition via recursive SVD

        Decomposes `input_tensor` into a sequence of order-3 tensors (factors) [1]_.

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : Union[int, List[int]]
            maximum allowable TR rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    mode : int, default is 0
            index of the first factor to compute
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    verbose : boolean, optional
            level of verbosity

    Returns
    -------
    factors : TR factors
              order-3 tensors of the TR decomposition

    References
    ----------
    .. [1] Qibin Zhao et al. "Tensor Ring Decomposition" arXiv preprint arXiv:1606.05535, (2016).
    """

    def __init__(self, rank, mode=0, svd="truncated_svd", verbose=False):
        self.rank = rank
        self.mode = mode
        self.svd = svd
        self.verbose = verbose

    def fit_transform(self, tensor):
        self.decomposition_ = tensor_ring(
            tensor, rank=self.rank, mode=self.mode, svd=self.svd, verbose=self.verbose
        )
        return self.decomposition_


class TensorRingALS(DecompositionMixin):
    """A class wrapper for the tensor_ring_als function

    Attributes
    ----------
    rank : Union[int, List[int]]
        The rank of the decomposition. If `rank` is an int, then all ranks will be the
        same and equal to `rank`. If `rank` is a list, then the i-th core will be of
        size rank[i]-by-shape[i]-by-rank[i+1], where shape[i] is the dimension of the
        i-th mode of `tensor`.
    ls_solve : {"lstsq", "normal_eq"}
        When equal to "lstsq", the least squares problems are solved as overdetermined
        systems of equations using tl.lstsq. When equal to "normal_eq", a normal
        equation formulation is used instead together with tl.solve. This latter
        approach is expected to yield poorer accuracy due to numerical issuse with the
        normal equations.
    n_iter_max : int
        Maximum number of ALS iterations.
    tol : float
        The algorithm is terminated when the change in the relative reconstruction error
        is less than `tol`.
    random_state : {None, int, np.random.RandomState}
        Used to set the random seed in the algorithm.
    verbose : bool
        If True, the algorithm will make some additional print outs.
    callback : {None, Callable[[tl.tr_tensor.TRTensor, float], {None, bool}]}
        A callback function which can be used for, e.g., logging the per-iteration
        error. Such a function takes two inputs: A tensor ring decomposition and a float
        which indicates the relative reconstruction error between `tensor` and the
        inputted TRTensor. It can return a bool which can be used to control when the
        ALS algorithm terminates.

    Methods
    -------
    fit_transformation(tensor)
        Computes the decomposition of `tensor`.
    """

    def __init__(
        self,
        rank,
        ls_solve="lstsq",
        n_iter_max=100,
        tol=1e-6,
        random_state=None,
        verbose=False,
        callback=None,
    ):
        """
        Parameters
        ----------
        rank : Union[int, List[int]]
            The rank of the decomposition. If `rank` is an int, then all ranks will be the
            same and equal to `rank`. If `rank` is a list, then the i-th core will be of
            size rank[i]-by-shape[i]-by-rank[i+1], where shape[i] is the dimension of the
            i-th mode of `tensor`.
        ls_solve : {"lstsq", "normal_eq"}, default is "lstsq"
            When equal to "lstsq", the least squares problems are solved as overdetermined
            systems of equations using tl.lstsq. When equal to "normal_eq", a normal
            equation formulation is used instead together with tl.solve. This latter
            approach is expected to yield poorer accuracy due to numerical issuse with the
            normal equations.
        n_iter_max : int, default is 100
            Maximum number of ALS iterations.
        tol : float, default 1e-6
            The algorithm is terminated when the change in the relative reconstruction error
            is less than `tol`.
        random_state : {None, int, np.random.RandomState}
            Used to set the random seed in the algorithm.
        verbose : bool, default False
            If True, the algorithm will make some additional print outs.
        callback : {None, Callable[[tl.tr_tensor.TRTensor, float], {None, bool}]}, default None
            A callback function which can be used for, e.g., logging the per-iteration
            error. Such a function takes two inputs: A tensor ring decomposition and a float
            which indicates the relative reconstruction error between `tensor` and the
            inputted TRTensor. It can return a bool which can be used to control when the
            ALS algorithm terminates.
        """

        self.rank = rank
        self.ls_solve = ls_solve
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.callback = callback

    def fit_transform(self, tensor):
        """Computes the decomposition of `tensor`.

        Parameters
        ----------
        tensor : ndarray

        Returns
        -------
        decomposition_ : tl.tr_tensor.TRTensor
            The tensor ring decomposition computed by the algorithm.
        """

        tr_decomp = tensor_ring_als(
            tensor,
            rank=self.rank,
            ls_solve=self.ls_solve,
            n_iter_max=self.n_iter_max,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            callback=self.callback,
        )
        self.decomposition_ = tr_decomp
        return self.decomposition_
