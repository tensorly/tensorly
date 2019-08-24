"""
Core operations on Kruskal tensors.
"""

from . import backend as T
from .base import fold, tensor_to_vec
from .tenalg import khatri_rao, multi_mode_dot, inner

import warnings
from collections.abc import Mapping

# Author: Jean Kossaifi

# License: BSD 3 clause

class KruskalTensor(Mapping):
    def __init__(self, kruskal_tensor):
        super().__init__()

        shape, rank = _validate_kruskal_tensor(kruskal_tensor)
        weights, factors = kruskal_tensor

        # Should we allow None weights?
        if weights is None:
            weights = T.ones(rank, **T.context(factors[0]))

        self.shape = shape
        self.rank = rank
        self.factors = factors
        self.weights = weights

    
    def __getitem__(self, index):
        if index == 0:
            return self.weights
        elif index == 1:
            return self.factors
        else: 
            raise IndexError('You tried to access index {} of a Kruskal tensor.\n'
                             'You can only access index 0 and 1 of a Kruskal tensor'
                             '(corresponding respectively to the weights and factors)'.format(index))
    
    def __iter__(self):
        yield self.weights
        yield self.factors
        
    def __len__(self):
        return 2
    
    def __repr__(self):
        message = '(weights, factors) : rank-{} KruskalTensor of shape {} '.format(self.rank, self.shape)
        return message


def _validate_kruskal_tensor(kruskal_tensor):
    """Validates a kruskal_tensor in the form (weights, factors)
    
        Returns the rank and shape of the validated tensor
    
    Parameters
    ----------
    kruskal_tensor : KruskalTensor or (weights, factors)
    
    Returns
    -------
    (shape, rank) : (int tuple, int)
        size of the full tensor and rank of the Kruskal tensor
    """
    if isinstance(kruskal_tensor, KruskalTensor):
        # it's already been validated at creation
        return kruskal_tensor.shape, kruskal_tensor.rank

    weights, factors = kruskal_tensor
            
    if len(factors) < 2:
        raise ValueError('A Kruskal tensor should be composed of at least two factors.'
                         'However, {} factor was given.'.format(len(factors)))

    if T.ndim(factors[0]) == 2:
        rank = int(T.shape(factors[0])[1])
    else:
        rank = 1
    shape = []
    for i, factor in enumerate(factors):
        s = T.shape(factor)
        if len(s) == 2:
            current_mode_size, current_rank = s
        else:
            current_mode_size, current_rank = s, 1

        if current_rank != rank:
            raise ValueError('All the factors of a Kruskal tensor should have the same number of column.'
                             'However, factors[0].shape[1]={} but factors[{}].shape[1]={}.'.format(
                                 rank, i, T.shape(factor)[1]))
        shape.append(current_mode_size)

    if weights is not None and len(weights) != rank:
        raise ValueError('Given factors for a rank-{} Kruskal tensor but len(weights)={}.'.format(
            rank, len(weights)))
        
    return tuple(shape), rank


def kruskal_normalise(kruskal_tensor):
    """Returns kruskal_tensor with factors normalised to unit length

    Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,
    where the columns of each `V_k` are normalized to unit Euclidean length
    from the columns of `U_k` with the normalizing constants absorbed into
    `weights`. In the special case of a symmetric tensor, `weights` holds the
    eigenvalues of the tensor.

    Parameters
    ----------
    kruskal_tensor : KruskalTensor = (weight, factors)
        factors is list of matrices, all with the same number of columns
        i.e.::
            for u in U:
                u[i].shape == (s_i, R)

        where `R` is fixed while `s_i` can vary with `i`

    Returns
    -------
    KruskalTensor = (normalisation_weights, normalised_factors)
    """
    _, rank = _validate_kruskal_tensor(kruskal_tensor)
    weights, factors = kruskal_tensor
    
    if weights is None:
        weights = T.ones(rank, **T.context(factors[0]))
    
    normalized_factors = []
    for i, factor in enumerate(factors):
        if i == 0:
            factor = factor*weights
            weights = T.ones(rank, **T.context(factor))
            
        scales = T.norm(factor, axis=0)
        scales_non_zero = T.where(scales==0, T.ones(T.shape(scales), **T.context(factor)), scales)
        weights = weights*scales
        normalized_factors.append(factor / T.reshape(scales_non_zero, (1, -1)))

    return KruskalTensor((weights, normalized_factors))
    

def kruskal_to_tensor(kruskal_tensor, mask=None):
    """Turns the Khatri-product of matrices into a full tensor

        ``factor_matrices = [|U_1, ... U_n|]`` becomes
        a tensor shape ``(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])``

    Parameters
    ----------
    kruskal_tensor : KruskalTensor = (weight, factors)
        factors is a list of factor matrices, all with the same number of columns
        i.e. for all matrix U in factor_matrices:
        U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

    mask : ndarray a mask to be applied to the final tensor. It should be
        broadcastable to the shape of the final tensor, that is
        ``(U[1].shape[0], ... U[-1].shape[0])``.

    Returns
    -------
    ndarray
        full tensor of shape ``(U[1].shape[0], ... U[-1].shape[0])``

    Notes
    -----
    This version works by first computing the mode-0 unfolding of the tensor
    and then refolding it.

    There are other possible and equivalent alternate implementation, e.g.
    summing over r and updating an outer product of vectors.

    """
    shape, rank = _validate_kruskal_tensor(kruskal_tensor)
    weights, factors = kruskal_tensor

    if weights is None:
        weights = 1

    if mask is None:
        full_tensor = T.dot(factors[0]*weights,
                             T.transpose(khatri_rao(factors, skip_matrix=0)))
    else:
        full_tensor = T.sum(khatri_rao([factor[0]*weights]+factors[1:], mask=mask), axis=1)

    return fold(full_tensor, 0, shape)

def kruskal_to_unfolded(kruskal_tensor, mode):
    """Turns the khatri-product of matrices into an unfolded tensor

        turns ``factors = [|U_1, ... U_n|]`` into a mode-`mode`
        unfolding of the tensor

    Parameters
    ----------
    kruskal_tensor : KruskalTensor = (weight, factors)
        factors is a list of matrices, all with the same number of columns
        ie for all u in factor_matrices:
        u[i] has shape (s_u_i, R), where R is fixed
    mode: int
        mode of the desired unfolding

    Returns
    -------
    ndarray
        unfolded tensor of shape (tensor_shape[mode], -1)

    Notes
    -----
    Writing factors = [U_1, ..., U_n], we exploit the fact that
    ``U_k = U[k].dot(khatri_rao(U_1, ..., U_k-1, U_k+1, ..., U_n))``
    """
    _validate_kruskal_tensor(kruskal_tensor)
    weights, factors = kruskal_tensor

    if weights is not None:
        return T.dot(factors[mode]*weights, T.transpose(khatri_rao(factors, skip_matrix=mode)))
    else:
        return T.dot(factors[mode], T.transpose(khatri_rao(factors, skip_matrix=mode)))


def kruskal_to_vec(kruskal_tensor):
    """Turns the khatri-product of matrices into a vector

        (the tensor ``factors = [|U_1, ... U_n|]``
        is converted into a raveled mode-0 unfolding)

    Parameters
    ----------
    kruskal_tensor : KruskalTensor = (weight, factors)
        factors is a list of matrices, all with the same number of columns
        i.e.::

            for u in U:
                u[i].shape == (s_i, R)

        where `R` is fixed while `s_i` can vary with `i`

    Returns
    -------
    ndarray
        vectorised tensor
    """
    return tensor_to_vec(kruskal_to_tensor(kruskal_tensor))


def kruskal_mode_dot(kruskal_tensor, matrix_or_vector, mode, keep_dim=False, copy=False):
        """n-mode product of a Kruskal tensor and a matrix or vector at the specified mode

        Parameters
        ----------
        kruskal_tensor : tl.KruskalTensor or (core, factors)
                        
        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        KruskalTensor = (core, factors)
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

        See also
        --------
        kruskal_multi_mode_dot : chaining several mode_dot in one call
        """
        shape, _ = _validate_kruskal_tensor(kruskal_tensor)
        weights, factors = kruskal_tensor
        contract = False
        
        if T.ndim(matrix_or_vector) == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                        shape, matrix_or_vector.shape, mode, shape[mode], matrix_or_vector.shape[1]
                    ))

        elif T.ndim(matrix_or_vector) == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != shape[mode]:
                raise ValueError(
                    'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                        shape, matrix_or_vector.shape, mode, shape[mode], matrix_or_vector.shape[0]
                    ))
            if not keep_dim:
                contract = True # Contract over that mode
        else:
            raise ValueError('Can only take n_mode_product with a vector or a matrix.')
                             
        if copy:
            factors = [T.copy(f) for f in factors]
            weights = T.copy(weights)   
            
        if contract:
            factor = factors.pop(mode)
            factor = T.dot(matrix_or_vector, factor)
            mode = max(mode - 1, 0)
            factors[mode] *= factor
        else:
             factors[mode] = T.dot(matrix_or_vector, factors[mode])

        return KruskalTensor((weights, factors))
    

def unfolding_dot_khatri_rao(tensor, kruskal_tensor, mode):
    """mode-n unfolding times khatri-rao product of factors
    
    Parameters
    ----------
    tensor : tl.tensor
        tensor to unfold
    factors : tl.tensor list
        list of matrices of which to the khatri-rao product
    mode : int
        mode on which to unfold `tensor`
    
    Returns
    -------
    mttkrp
        dot(unfold(tensor, mode), khatri-rao(factors))

    Notes
    -----
    This is a variant of::
    
        unfolded = unfold(tensor, mode)
        kr_factors = khatri_rao(factors, skip_matrix=mode)
        mttkrp2 = tl.dot(unfolded, kr_factors)

    Multiplying with the Khatri-Rao product is equivalent to multiplying,
    for each rank, with the kronecker product of each factor. 
    In code::

        mttkrp_parts = []
        for r in range(rank):
            component = tl.tenalg.multi_mode_dot(tensor, [f[:, r] for f in factors], skip=mode)
            mttkrp_parts.append(component)
        mttkrp = tl.stack(mttkrp_parts, axis=1)
        return mttkrp 

    This can be done by taking n-mode-product with the full factors 
    (faster but more memory consuming)::

        projected = multi_mode_dot(tensor, factors, skip=mode, transpose=True)
        ndims = T.ndim(tensor)
        res = []
        for i in range(factors[0].shape[1]):
            index = tuple([slice(None) if k == mode  else i for k in range(ndims)])
            res.append(projected[index])
        return T.stack(res, axis=-1)

    
    The same idea could be expressed using einsum::
    
        ndims = tl.ndim(tensor)
        tensor_idx = ''.join(chr(ord('a') + i) for i in range(ndims))
        rank = chr(ord('a') + ndims + 1)
        op = tensor_idx
        for i in range(ndims):
            if i != mode:
                op += ',' + ''.join([tensor_idx[i], rank])
            else:
                result = ''.join([tensor_idx[i], rank])
        op += '->' + result
        factors = [f for (i, f) in enumerate(factors) if i != mode]
        return tl_einsum(op, tensor, *factors)
    """
    mttkrp_parts = []
    _, rank = _validate_kruskal_tensor(kruskal_tensor)
    weights, factors = kruskal_tensor
    for r in range(rank):
        component = multi_mode_dot(tensor, [f[:, r] for f in factors], skip=mode)
        mttkrp_parts.append(component)

    if weights is None:
        return T.stack(mttkrp_parts, axis=1)
    else:
        return T.stack(mttkrp_parts, axis=1)*T.reshape(weights, (1, -1))


def kruskal_norm(kruskal_tensor):
    """Returns the l2 norm of a Kruskal tensor

    Parameters
    ----------
    kruskal_tensor : tl.KruskalTensor or (core, factors)

    Returns
    -------
    l2-norm : int

    Notes
    -----
    This is ||kruskal_to_tensor(factors)||^2 
    
    You can see this using the fact that
    khatria-rao(A, B)^T x khatri-rao(A, B) = A^T x A  * B^T x B
    """
    _ = _validate_kruskal_tensor(kruskal_tensor)
    weights, factors = kruskal_tensor
    norm = 1
    for factor in factors:
        norm *= T.dot(T.transpose(factor), factor)
    
    if weights is not None:
        #norm = T.dot(T.dot(weights, norm), weights)
        norm = norm * (T.reshape(weights, (-1, 1))*T.reshape(weights, (1, -1)))

    # We sum even if weigths is not None
    # as e.g. MXNet would return a 1D tensor, not a 0D tensor
    return T.sqrt(T.sum(norm))
