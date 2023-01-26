import numpy as np
import tensorly as tl
from tensorly.tt_tensor import tt_to_tensor, validate_tt_rank
from tensorly.decomposition._base_decomposition import DecompositionMixin


def sequential_prod(tensor_prod, multiplier_list, left_to_right=True):
    """Perform sequential multiplication and reshaping

    Parameters
    ----------
    tensor_prod: ndrray
        the tensor to be multiplied by a list of tensors
        tensor_prod is of dimension r_0*p_1*p_2*...*p_d*r_{d-1}
    multiplier_list: list
        a list of tensors to multiply the tensor
        len(multiplier_list) <= d-1
        If direction == "left", multiplier_list[i] is of shape (r_i, p_{i+1}, r_{i+1})
        If direction == "right", multiplier_list[i] is of shape (r_{d-i}, p_{d-i}, r_{d-i-1})
    left_to_right : bool, optional, default is True
        direction of the multiplication
    Returns
    -------
    tensor_prod : ndarray
        product tensor
        If direction == "left" and `multiplier_list` contains N tensors, then `tensor_prod` is of dimension r_N*p_{N+1}*...*p_d*r_{d-1}
        If direction == "right" and `multiplier_list` contains d-N tensors, then `tensor_prod` is of dimension r_0*p_1*...*p_N*r_N
    """

    if left_to_right == True:
        for i in range(len(multiplier_list)):
            tensor_prod = tl.tensordot(
                multiplier_list[i], tensor_prod, axes=([0, 1], [0, 1])
            )
    else:
        for i in range(len(multiplier_list)):
            tensor_prod = tl.tensordot(
                tensor_prod,
                multiplier_list[i],
                axes=([tl.ndim(tensor_prod) - 1, tl.ndim(tensor_prod) - 2], [0, 1]),
            )
    return tensor_prod


def tensor_train_OI(data_tensor, rank, n_iter=2, trajectory=False, return_errors=True):
    """Perform tensor-train orthogonal iteration (TTOI) [1]_ for tensor train decomposition

    Parameters
    ----------
    data_tensor: tl.tensor
        observed tensor data
    rank : tuple
        rank of the TT decomposition
        must verify rank[0] == rank[-1] == 1 (boundary conditions)
        and len(rank) == len(tl.shape(data_tensor))+1
    n_iter : int
        the number of iterations
    trajectory : bool, optional, default is False
        if True, the output of each iteration of TTOI is returned: 2*n_iter outputs
        otherwise, only the output of the last iteration is returned
    return_errors : bool, optional, default is True
        if True, the approximation/reconstruction error of each iteration of TTOI is returned: 2*n_iter outputs

    Returns
    -------
    factors : list of n_iter lists of tensor-train factors or one list of tensor-train factors::

        * n_iter lists of factors (if `trajectory` is True) : each list contains the returned list of tensor-train factors from each iteration
        * one list of factors (otherwise): returned list of tensor-train factors from the last iteration

    full_tensor : list of n_iter tensors or one tensor::

        * n_iter tensors (if `trajectory` is True) : each list contains the returned full tensor from each iteration
        * one tensor (otherwise): returned full tensor from the last iteration

    References
    ----------
    .. [1] Zhou, Y., Zhang, A. R., Zheng, L., & Wang, Y. (2022). Optimal high-order tensor svd via tensor-train orthogonal iteration.
    IEEE Transactions on Information Theory, 68(6), 3991-4019.
    """
    context = tl.context(data_tensor)
    shape = tl.shape(data_tensor)
    n_dim = len(shape)

    rank = validate_tt_rank(shape, rank)

    # Make sure it's not a tuple but a list
    rank = list(rank)

    # Add two one-dimensional mode to data_tensor
    data_tensor_extended = tl.reshape(data_tensor, (1,) + shape + (1,))

    if trajectory:
        factors = []
        full_tensor = []

    if return_errors:
        error_list = []

    # perform TTOI for n_iter iterations
    for iteration in range(int(np.ceil(n_iter / 2))):
        # first perform forward update
        # left_singular_vectors will be a list including estimated left singular spaces at the current iteration
        left_singular_vectors = []
        # initialize left_residuals (sequential unfolding of data_tensor multiplied by left_singular_vectors sequentially on the left, useful for backward update to obtain right_singular_vectors)
        left_residuals = []

        # estimate the first left singular spaces
        # Here, R_tmp is the first sequential unfolding compressed on the right by previous updated right_singular_vectors (if exists)
        R_tmp_l = data_tensor_extended
        if iteration == 0:
            R_tmp = R_tmp_l
        else:
            R_tmp = sequential_prod(
                R_tmp_l, right_singular_vectors, left_to_right=False
            )
        U_tmp, _, _ = tl.tenalg.svd_interface(
            matrix=tl.reshape(R_tmp, (shape[0], -1)), n_eigenvecs=rank[1]
        )
        left_singular_vectors.append(tl.reshape(U_tmp, (rank[0], shape[0], rank[1])))

        # estimate the 2nd to (d-1)th left singular spaces
        for mode in range(n_dim - 2):
            # compress the (mode+2)th sequential unfolding of data_tensor from the left
            R_tmp_l = sequential_prod(
                R_tmp_l, [left_singular_vectors[mode]], left_to_right=True
            )
            # R_tmp_l will be useful for backward update
            left_residuals.append(R_tmp_l)

            # compress the (mode+2)th sequential unfolding of data_tensor from the right (if iteration>0)
            if iteration == 0:
                R_tmp = R_tmp_l
            else:
                R_tmp = sequential_prod(
                    R_tmp_l,
                    right_singular_vectors[0 : (n_dim - mode - 2)],
                    left_to_right=False,
                )
            U_tmp, _, _ = tl.tenalg.svd_interface(
                matrix=tl.reshape(R_tmp, (rank[mode + 1] * shape[mode + 1], -1)),
                n_eigenvecs=rank[mode + 2],
            )
            left_singular_vectors.append(
                tl.reshape(U_tmp, (rank[mode + 1], shape[mode + 1], rank[mode + 2]))
            )

        # forward update is done; output the final residual
        left_residuals.append(
            sequential_prod(
                R_tmp_l, [left_singular_vectors[n_dim - 2]], left_to_right=True
            )
        )
        if trajectory or return_errors:
            factors_list_tmp = []
            for mode in range(n_dim - 1):
                factors_list_tmp.append(left_singular_vectors[mode])
            factors_list_tmp.append(left_residuals[n_dim - 2])
            full_tensor_tmp = tt_to_tensor(factors_list_tmp)
            if return_errors:
                error_list.append(tl.norm(full_tensor_tmp - data_tensor, 2))
            if trajectory:
                factors.append(factors_list_tmp)
                full_tensor.append(full_tensor_tmp)
            elif (
                iteration == np.ceil(n_iter / 2) - 1
                and np.ceil(n_iter / 2) * 2 > n_iter
            ):
                factors = factors_list_tmp
                full_tensor = full_tensor_tmp

        if np.ceil(n_iter / 2) * 2 == n_iter:
            # perform backward update
            # initialize right_singular_vectors: right_singular_vectors will be a list of estimated right singular spaces at the current or previous iteration
            right_singular_vectors = []
            _, _, V_tmp = tl.tenalg.svd_interface(
                matrix=tl.reshape(
                    left_residuals[n_dim - 2], (rank[n_dim - 1], shape[n_dim - 1])
                ),
                n_eigenvecs=rank[n_dim - 1],
            )
            V_tmp = tl.transpose(V_tmp)
            right_singular_vectors.append(
                tl.reshape(V_tmp, (rank[n_dim], shape[n_dim - 1], rank[n_dim - 1]))
            )
            # estimate the 2nd to (d-1)th right singular spaces
            for mode in range(n_dim - 2):
                # compress left_residuals from the right
                R_tmp_r = sequential_prod(
                    left_residuals[n_dim - mode - 3],
                    right_singular_vectors[0 : (mode + 1)],
                    "right",
                )
                _, _, V_tmp = tl.tenalg.svd_interface(
                    matrix=tl.reshape(
                        R_tmp_r,
                        (
                            rank[n_dim - mode - 2],
                            shape[n_dim - mode - 2] * rank[n_dim - mode - 1],
                        ),
                    ),
                    n_eigenvecs=rank[n_dim - mode - 2],
                )
                right_singular_vectors.append(
                    tl.transpose(
                        tl.reshape(
                            V_tmp,
                            (
                                rank[n_dim - mode - 2],
                                shape[n_dim - mode - 2],
                                rank[n_dim - mode - 1],
                            ),
                        )
                    )
                )
            Residual_right = sequential_prod(
                data_tensor_extended, right_singular_vectors, left_to_right=False
            )
            if trajectory or return_errors or iteration == np.ceil(n_iter / 2) - 1:
                factors_list_tmp = []
                factors_list_tmp.append(Residual_right)
                for mode in range(n_dim - 1):
                    factors_list_tmp.append(
                        tl.transpose(right_singular_vectors[n_dim - mode - 2])
                    )
                full_tensor_tmp = tt_to_tensor(factors_list_tmp)
                if return_errors:
                    error_list.append(tl.norm(full_tensor_tmp - data_tensor, 2))
                if trajectory:
                    factors.append(factors_list_tmp)
                    full_tensor.append(full_tensor_tmp)
                elif iteration == n_iter - 1:
                    factors = factors_list_tmp
                    full_tensor = full_tensor_tmp

    # return final results
    if return_errors:
        return factors, full_tensor, error_list
    else:
        return factors, full_tensor


class TensorTrain_OI(DecompositionMixin):
    def __init__(self, rank, n_iter, trajectory, return_errors):
        self.rank = rank
        self.n_iter = n_iter
        self.trajectory = trajectory
        self.return_errors = return_errors

    def fit_transform(self, tensor):
        self.decomposition_ = tensor_train_OI(
            tensor,
            rank=self.rank,
            n_iter=self.n_iter,
            trajectory=self.trajectory,
            return_errors=self.return_errors,
        )
        return self.decomposition_
