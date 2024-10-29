"""
Core operations on tensors in Tensor-Train (TT) format, also known as Matrix-Product-State (MPS)
"""

import tensorly as tl
from ._factorized_tensor import FactorizedTensor
import numpy as np
import warnings


def _validate_tt_tensor(tt_tensor):
    factors = tt_tensor
    n_factors = len(factors)

    if isinstance(tt_tensor, TTTensor):
        # it's already been validated at creation
        return tt_tensor.shape, tt_tensor.rank
    elif isinstance(tt_tensor, (float, int)):  # 0-order tensor
        return 0, 0

    rank = []
    shape = []
    for index, factor in enumerate(factors):
        current_rank, current_shape, next_rank = tl.shape(factor)

        # Check that factors are third order tensors
        if not tl.ndim(factor) == 3:
            raise ValueError(
                "TT expresses a tensor as third order factors (tt-cores).\n"
                f"However, tl.ndim(factors[{index}]) = {tl.ndim(factor)}"
            )
        # Consecutive factors should have matching ranks
        if index and tl.shape(factors[index - 1])[2] != current_rank:
            raise ValueError(
                "Consecutive factors should have matching ranks\n"
                " -- e.g. tl.shape(factors[0])[2]) == tl.shape(factors[1])[0])\n"
                f"However, tl.shape(factor[{index-1}])[2] == {tl.shape(factors[index - 1])[2]} but"
                f" tl.shape(factor[{index}])[0] == {current_rank} "
            )
        # Check for boundary conditions
        if (index == 0) and current_rank != 1:
            raise ValueError(
                "Boundary conditions dictate factor[0].shape[0] == 1."
                f"However, got factor[0].shape[0] = {current_rank}."
            )
        if (index == n_factors - 1) and next_rank != 1:
            raise ValueError(
                "Boundary conditions dictate factor[-1].shape[2] == 1."
                f"However, got factor[{n_factors}].shape[2] = {next_rank}."
            )

        shape.append(current_shape)
        rank.append(current_rank)

    # Add last rank (boundary condition)
    rank.append(next_rank)

    return tuple(shape), tuple(rank)


def tt_to_tensor(factors):
    """Returns the full tensor whose TT decomposition is given by 'factors'

        Re-assembles 'factors', which represent a tensor in TT/Matrix-Product-State format
        into the corresponding full tensor

    Parameters
    ----------
    factors : list of 3D-arrays
              TT factors (TT-cores)

    Returns
    -------
    output_tensor : ndarray
                   tensor whose TT/MPS decomposition was given by 'factors'
    """
    if isinstance(factors, (float, int)):  # 0-order tensor
        return factors

    full_shape = [f.shape[1] for f in factors]
    full_tensor = tl.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return tl.reshape(full_tensor, full_shape)


def tt_to_unfolded(factors, mode):
    """Returns the unfolding matrix of a tensor given in TT (or Tensor-Train) format

    Reassembles a full tensor from 'factors' and returns its unfolding matrix
    with mode given by 'mode'

    Parameters
    ----------
    factors: list of 3D-arrays
              TT factors
    mode: int
          unfolding matrix to be computed along this mode

    Returns
    -------
    2-D array
    unfolding matrix at mode given by 'mode'
    """
    return tl.unfold(tt_to_tensor(factors), mode)


def tt_to_vec(factors):
    """Returns the tensor defined by its TT format ('factors') into
       its vectorized format

    Parameters
    ----------
    factors: list of 3D-arrays
              TT factors

    Returns
    -------
    1-D array
    vectorized format of tensor defined by 'factors'
    """
    return tl.tensor_to_vec(tt_to_tensor(factors))


def _tt_n_param(tensor_shape, rank):
    """Number of parameters of a MPS decomposition for a given `rank` and full `tensor_shape`.

    Parameters
    ----------
    tensor_shape : int tuple
        shape of the full tensor to decompose (or approximate)

    rank : tuple
        rank of the MPS decomposition

    Returns
    -------
    n_params : int
        Number of parameters of a MPS decomposition of rank `rank` of a full tensor of shape `tensor_shape`
    """
    factor_params = []
    for i, s in enumerate(tensor_shape):
        factor_params.append(rank[i] * s * rank[i + 1])
    return np.sum(factor_params)


def validate_tt_rank(
    tensor_shape,
    rank="same",
    constant_rank=False,
    rounding="round",
    allow_overparametrization=True,
):
    """Returns the rank of a TT Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, tuple, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple, just returns rank
    constant_rank : bool, default is False
        * if True, the *same* rank will be chosen for each modes
        * if False (default), the rank of each mode will be proportional to the corresponding tensor_shape

        *used only if rank == 'same' or 0 < rank <= 1*

    rounding = {'round', 'floor', 'ceil'}

    allow_overparametrization : bool, default is True
        if False, the rank must be realizable through iterative application of SVD
        (used in tensorly.decomposition.tensor_train)

    Returns
    -------
    rank : int tuple
        rank of the decomposition
    """
    if rounding == "ceil":
        rounding_fun = np.ceil
    elif rounding == "floor":
        rounding_fun = np.floor
    elif rounding == "round":
        rounding_fun = np.round
    else:
        raise ValueError(f"Rounding should be round, floor or ceil, but got {rounding}")

    if rank == "same":
        rank = float(1)

    if isinstance(rank, float) and constant_rank:
        # Choose the *same* rank for each mode
        n_param_tensor = np.prod(tensor_shape) * rank
        order = len(tensor_shape)

        if order == 2:
            rank = (1, n_param_tensor / (tensor_shape[0] + tensor_shape[1]), 1)
            warnings.warn(
                f"Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor."
            )

        # R_k I_k R_{k+1} = R^2 I_k
        a = np.sum(tensor_shape[1:-1])

        # Border rank of 1, R_0 = R_N = 1
        # First and last factor of size I_0 R and I_N R
        b = np.sum(tensor_shape[0] + tensor_shape[-1])

        # We want the number of params of decomp (=sum of params of factors)
        # To be equal to c = \prod_k I_k
        c = -n_param_tensor
        delta = np.sqrt(b**2 - 4 * a * c)

        # We get the non-negative solution
        solution = int(rounding_fun((-b + delta) / (2 * a)))
        rank = rank = (1,) + (solution,) * (order - 1) + (1,)

    elif isinstance(rank, float):
        # Choose a rank proportional to the size of each mode
        # The method is similar to the above one for constant_rank == True
        order = len(tensor_shape)
        avg_dim = [
            (tensor_shape[i] + tensor_shape[i + 1]) / 2 for i in range(order - 1)
        ]
        if len(avg_dim) > 1:
            a = sum(
                avg_dim[i - 1] * tensor_shape[i] * avg_dim[i]
                for i in range(1, order - 1)
            )
        else:
            warnings.warn(
                f"Determining the tt-rank for the trivial case of a matrix (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor."
            )
            a = avg_dim[0] ** 2 * tensor_shape[0]
        b = tensor_shape[0] * avg_dim[0] + tensor_shape[-1] * avg_dim[-1]
        c = -np.prod(tensor_shape) * rank
        delta = np.sqrt(b**2 - 4 * a * c)

        # We get the non-negative solution
        fraction_param = (-b + delta) / (2 * a)
        rank = tuple([max(int(rounding_fun(d * fraction_param)), 1) for d in avg_dim])
        rank = (1,) + rank + (1,)

    else:
        # Check user input for potential errors
        n_dim = len(tensor_shape)
        if isinstance(rank, int):
            rank = [1] + [rank] * (n_dim - 1) + [1]
        elif n_dim + 1 != len(rank):
            message = f"Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {len(rank)} while tl.ndim(tensor) + 1  = {n_dim+1}"
            raise (ValueError(message))

        # Initialization
        if rank[0] != 1:
            message = f"Provided rank[0] == {rank[0]} but boundary conditions dictate rank[0] == rank[-1] == 1."
            raise ValueError(message)
        if rank[-1] != 1:
            message = f"Provided rank[-1] == {rank[-1]} but boundary conditions dictate rank[0] == rank[-1] == 1."
            raise ValueError(message)

    if allow_overparametrization:
        return list(rank)
    else:
        validated_rank = [1]
        for i, s in enumerate(tensor_shape[:-1]):
            n_row = int(rank[i] * s)
            n_column = np.prod(tensor_shape[(i + 1) :])  # n_column of unfolding
            validated_rank.append(min(n_row, n_column, rank[i + 1]))
        validated_rank.append(1)

        return validated_rank


class TTTensor(FactorizedTensor):
    def __init__(self, factors, inplace=False):
        super().__init__()

        # Will raise an error if invalid
        shape, rank = _validate_tt_tensor(factors)

        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors

    def __getitem__(self, index):
        return self.factors[index]

    def __setitem__(self, index, value):
        self.factors[index] = value

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __len__(self):
        return len(self.factors)

    def __repr__(self):
        message = f"factors list : rank-{self.rank} matrix-product-state tensor of shape {self.shape} "
        return message

    def to_tensor(self):
        return tt_to_tensor(self)

    def to_unfolding(self, mode):
        return tt_to_unfolded(self, mode)

    def to_vec(self):
        return tt_to_vec(self)


def pad_tt_rank(factor_list, n_padding=1, pad_boundaries=False):
    """Pads the factors of a Tensor-Train so as to increase its rank without changing its reconstruction

    The tensor-train (ring) will be padded with 0s to increase its rank only but not the underlying tensor it represents.

    Parameters
    ----------
    factor_list : tensor list
    n_padding : int, default is 1
        how much to increase the rank (bond dimension) by
    pad_boundaries : bool, default is False
        if True, also pad the boundaries (useful for a tensor-ring)
        should be False for a tensor-train to keep the boundary rank to be 1

    Returns
    -------
    padded_factor_list
    """
    new_factors = []
    n_factors = len(factor_list)

    for i, factor in enumerate(factor_list):
        n_padding_left = n_padding_right = n_padding
        if (i == 0) and not pad_boundaries:
            n_padding_left = 0
        elif (i == n_factors - 1) and not pad_boundaries:
            n_padding_right = 0

        r1, *s, r2 = tl.shape(factor)
        new_factor = tl.zeros(
            (r1 + n_padding_left, *s, r2 + n_padding_right), **tl.context(factor)
        )
        new_factors.append(tl.index_update(new_factor, tl.index[:r1, ..., :r2], factor))

    return new_factors
