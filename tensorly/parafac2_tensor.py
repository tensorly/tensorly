"""Core operations on PARAFAC2 tensors whose second mode evolve over their first.
"""

# Authors: Marie Roald
#          Yngve Mardal Moe

from . import backend as T
from .base import unfold, tensor_to_vec
from ._factorized_tensor import FactorizedTensor
import warnings

class Parafac2Tensor(FactorizedTensor):
    """A wrapper class for the PARAFAC2 decomposition.
    """
    def __init__(self, parafac2_tensor):

        super().__init__()

        shape, rank = _validate_parafac2_tensor(parafac2_tensor)
        weights, factors, projections = parafac2_tensor

        if weights is None:
            weights = T.ones(rank, **T.context(factors[0]))

        self.shape = shape
        self.rank = rank
        self.factors = factors
        self.weights = weights
        self.projections = projections
        
    @classmethod
    def from_CPTensor(self, cp_tensor, parafac2_tensor_ok=False):
        """Create a Parafac2Tensor from a CPTensor

        Parameters:
        -----------
        cp_tensor: CPTensor or Parafac2Tensor
            If it is a Parafac2Tensor, then the argument ``parafac2_tensor_ok`` must be True'
        parafac2_tensor: bool (optional)
            Whether or not Parafac2Tensors can be used as input.

        Returns:
        --------
        Parafac2Tensor
            Parafac2Tensor with factor matrices and weigths extracted from a CPTensor
        """
        if parafac2_tensor_ok and len(cp_tensor) == 3:
            return Parafac2Tensor(cp_tensor)
        elif len(cp_tensor) == 3:
            raise TypeError('Input is not a CPTensor. If it is a Parafac2Tensor, then the argument ``parafac2_tensor_ok`` must be True')
        
        weights, (A, B, C) = cp_tensor
        Q, R = T.qr(B)
        projections = [Q for _ in range(T.shape(A)[0])]
        B = R
        return Parafac2Tensor((weights, (A, B, C), projections))

    def __getitem__(self, index):
        if index == 0:
            return self.weights
        elif index == 1:
            return self.factors
        elif index ==2:
            return self.projections
        else: 
            raise IndexError('You tried to access index {} of a PARAFAC2 tensor.\n'
                             'You can only access index 0, 1 and 2 of a PARAFAC2 tensor'
                             '(corresponding respectively to the weights, factors and projections)'.format(index))
    
    def __iter__(self):
        yield self.weights
        yield self.factors
        yield self.projections
        
    def __len__(self):
        return 3
    
    def __repr__(self):
        message = '(weights, factors, projections) : rank-{} Parafac2Tensor of shape {} '.format(self.rank, self.shape)
        return message

    def to_tensor(self):
        return parafac2_to_tensor(self)

    def to_vec(self):
        return parafac2_to_vec(self)
    
    def to_unfolded(self, mode):
        return parafac2_to_unfolded(self, mode)


def _validate_parafac2_tensor(parafac2_tensor):
    """Validates a parafac2_tensor in the form (weights, factors)
    
        Returns the rank and shape of the validated tensor
    
    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor or (weights, factors)
    
    Returns
    -------
    (shape, rank) : (int tuple, int)
        size of the full tensor and rank of the Kruskal tensor
    """
    if isinstance(parafac2_tensor, Parafac2Tensor):
        # it's already been validated at creation
        return parafac2_tensor.shape, parafac2_tensor.rank

    weights, factors, projections = parafac2_tensor
            
    if len(factors) != 3:
        raise ValueError('A PARAFAC2 tensor should be composed of exactly three factors.'
                         'However, {} factors was given.'.format(len(factors)))

    if len(projections) != factors[0].shape[0]:
        raise ValueError('A PARAFAC2 tensor should have one projection matrix for each horisontal'
                         ' slice. However, {} projection matrices was given and the first mode has'
                         'length {}'.format(len(projections), factors[0].shape[0]))

    rank = int(T.shape(factors[0])[1])

    shape = []
    for i, projection in enumerate(projections):
        current_mode_size, current_rank = T.shape(projection)
        if current_rank != rank:
            raise ValueError(
                'All the projection matrices of a PARAFAC2 tensor should have the same number of '
                'columns as the rank. However, rank={} but projections[{}].shape[1]={}'.format(
                    rank, i, T.shape(projection)[1]
                )
            )

        inner_product = T.dot(T.transpose(projection), projection)
        if T.max(T.abs(inner_product - T.eye(rank, **T.context(inner_product)))) > 1e-5:
            raise ValueError(
                'All the projection matrices must be orthonormal, that is, P.T@P = I. '
                'However, T.norm(projection[{}].T@projection[{}] - T.eye(rank)) = {}'.format(
                    i, i, T.norm(inner_product - T.eye(rank, **T.context(inner_product))) 
                )
            )
        
        shape.append((current_mode_size, *[f.shape[0] for f in factors[2:]]))  # Tuple unpacking to possibly support higher order PARAFAC2 tensors in the future

    
    # Skip first factor matrix since the rank is extracted from it.
    for i, factor in enumerate(factors[1:]):
        current_mode_size, current_rank = T.shape(factor)
        if current_rank != rank:
            raise ValueError('All the factors of a PARAFAC2 tensor should have the same number of columns.'
                             'However, factors[0].shape[1]={} but factors[{}].shape[1]={}.'.format(
                                 rank, i, T.shape(factor)[1]))

    if weights is not None and T.shape(weights)[0]  != rank:
        raise ValueError('Given factors for a rank-{} PARAFAC2 tensor but len(weights)={}.'.format(
            rank, T.shape(weights)[0]))
        
    return tuple(shape), rank


def parafac2_normalise(parafac2_tensor):
    """Returns parafac2_tensor with factors normalised to unit length

    Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,
    where the columns of each `V_k` are normalized to unit Euclidean length
    from the columns of `U_k` with the normalizing constants absorbed into
    `weights`. In the special case of a symmetric tensor, `weights` holds the
    eigenvalues of the tensor.

    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor = (weight, factors, projections)
        factors is list of matrices, all with the same number of columns
        i.e.::
            for u in U:
                u[i].shape == (s_i, R)

        where `R` is fixed while `s_i` can vary with `i`

    Returns
    -------
    Parafac2Tensor = (normalisation_weights, normalised_factors, normalised_projections)
    """
    # allocate variables for weights, and normalized factors
    _, rank = _validate_parafac2_tensor(parafac2_tensor)
    weights, factors, projections = parafac2_tensor
    
    # if (not copy) and (weights is None):
    #     warnings.warn('Provided copy=False and weights=None: a new Parafac2Tensor'
    #                   'with new weights and factors normalised inplace will be returned.')
    #     weights = T.ones(rank, **T.context(factors[0]))
    
    # The if test below was added to enable inplace edits
    # however, TensorFlow does not support inplace edits
    # so this is always set to True
    if True:  
        factors = [T.copy(f) for f in factors]
        projections = [T.copy(p) for p in projections]
        if weights is not None:
            factors[0] = factors[0] * weights
        weights = T.ones(rank, **T.context(factors[0]))
        
    for i, factor in enumerate(factors):
        scales = T.norm(factor, axis=0)
        weights = weights*scales
        scales_non_zero = T.where(scales==0, T.ones(T.shape(scales), **T.context(factors[0])), scales)
        factors[i] = factor/scales_non_zero
        
    return Parafac2Tensor((weights, factors, projections))


def apply_parafac2_projections(parafac2_tensor):
    """Apply the projection matrices to the evolving factor.
    
    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor
        
    Returns
    -------
    (weights, factors) : ndarray, tuple
        A tensor decomposition on the form A [B_i] C such that
        the :math:`X_{ijk}` is given by :math:`\sum_r A_{ir} [B_i]_{jr} C_{kr}`.

        This is also equivalent to a coupled matrix factorisation, where
        each matrix, :math:`X_i = C diag([a_{i1}, ..., a_{ir}] B_i)`.

        The first element of factors is the A matrix, the second element is
        a list of B-matrices and the third element is the C matrix.
    """
    _validate_parafac2_tensor(parafac2_tensor)
    weights, factors, projections = parafac2_tensor

    evolving_factor = [T.dot(projection, factors[1]) for projection in projections]
    
    return weights, (factors[0], evolving_factor, factors[2])


def parafac2_to_slice(parafac2_tensor, slice_idx, validate=True):
    r"""Generate a single slice along the first mode from the PARAFAC2 tensor.

    The decomposition is on the form :math:`(A [B_i] C)` such that the i-th frontal slice,
    :math:`X_i`, of :math:`X` is given by

    .. math::
    
        X_i = B_i diag(a_i) C^T,
    
    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i` 
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix. 
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.

    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},
    
    with the same constraints hold for :math:`B_i` as above.

    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the PARAFAC2 decomposition
            Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
        * projection_matrices : List of projection matrices used to create evolving
            factors.
    
    Returns
    -------
    ndarray
        Full tensor of shape [P[slice_idx].shape[1], C.shape[1]], where
        P is the projection matrices and C is the last factor matrix of
        the Parafac2Tensor.
    """
    if validate:
        _validate_parafac2_tensor(parafac2_tensor)
    weights, (A, B, C), projections = parafac2_tensor
    a = A[slice_idx]
    if weights is not None:
        a = a*weights

    Ct = T.transpose(C)

    B_i = T.dot(projections[slice_idx], B)
    return T.dot(B_i*a, Ct)


def parafac2_to_slices(parafac2_tensor, validate=True):
    r"""Generate all slices along the first mode from a PARAFAC2 tensor.

    Generates a list of all slices from a PARAFAC2 tensor. A list is returned
    since the tensor might have varying size along the second mode. To return
    a tensor, see the ``parafac2_to_tensor`` function instead.shape

    The decomposition is on the form :math:`(A [B_i] C)` such that the i-th frontal slice,
    :math:`X_i`, of :math:`X` is given by

    .. math::
    
        X_i = B_i diag(a_i) C^T,
    
    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i` 
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix. 
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.

    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},
    
    with the same constraints hold for :math:`B_i` as above.

    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the PARAFAC2 decomposition
            Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
        * projection_matrices : List of projection matrices used to create evolving
            factors.
    
    Returns
    -------
    List[ndarray]
        A list of full tensors of shapes [P[i].shape[1], C.shape[1]], where
        P is the projection matrices and C is the last factor matrix of the
        Parafac2Tensor.
    """
    if validate:
        _validate_parafac2_tensor(parafac2_tensor)
    weights, (A, B, C), projections = parafac2_tensor
    if weights is not None:
        A = A*weights
        weights = None

    decomposition = weights, (A, B, C), projections
    I, _ = A.shape
    return [parafac2_to_slice(decomposition, i, validate=False) for i in range(I)]


def parafac2_to_tensor(parafac2_tensor):
    r"""Construct a full tensor from a PARAFAC2 decomposition.

    The decomposition is on the form :math:`(A [B_i] C)` such that the i-th frontal slice,
    :math:`X_i`, of :math:`X` is given by

    .. math::
    
        X_i = B_i diag(a_i) C^T,
    
    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i` 
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix. 
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.

    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},
    
    with the same constraints hold for :math:`B_i` as above.

    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the PARAFAC2 decomposition
            Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
        * projection_matrices : List of projection matrices used to create evolving
            factors.
    
    Returns
    -------
    ndarray
        Full constructed tensor. Uneven slices are padded with zeros.
    """
    _, (A, _, C), projections = parafac2_tensor
    slices = parafac2_to_slices(parafac2_tensor)
    lengths = [projection.shape[0] for projection in projections]
    
    tensor = T.zeros((A.shape[0], max(lengths), C.shape[0]),  **T.context(slices[0]))
    for i, (slice_, length) in enumerate(zip(slices, lengths)):
        tensor = T.index_update(tensor, T.index[i, :length], slice_)
    return tensor


def parafac2_to_unfolded(parafac2_tensor, mode):
    r"""Construct an unfolded tensor from a PARAFAC2 decomposition. Uneven slices are padded by zeros.

    The decomposition is on the form :math:`(A [B_i] C)` such that the i-th frontal slice,
    :math:`X_i`, of :math:`X` is given by

    .. math::
    
        X_i = B_i diag(a_i) C^T,
    
    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i` 
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix. 
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.

    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},
    
    with the same constraints hold for :math:`B_i` as above.

    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the PARAFAC2 decomposition
            Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
        * projection_matrices : List of projection matrices used to create evolving
            factors.
    
    Returns
    -------
    ndarray
        Full constructed tensor. Uneven slices are padded with zeros.
    """
    return unfold(parafac2_to_tensor(parafac2_tensor), mode)


def parafac2_to_vec(parafac2_tensor):
    r"""Construct a vectorized tensor from a PARAFAC2 decomposition. Uneven slices are padded by zeros.

    The decomposition is on the form :math:`(A [B_i] C)` such that the i-th frontal slice,
    :math:`X_i`, of :math:`X` is given by

    .. math::
    
        X_i = B_i diag(a_i) C^T,
    
    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i` 
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix. 
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.

    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},
    
    with the same constraints hold for :math:`B_i` as above.

    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the PARAFAC2 decomposition
            Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
        * projection_matrices : List of projection matrices used to create evolving
            factors.
    
    Returns
    -------
    ndarray
        Full constructed tensor. Uneven slices are padded with zeros.
    """
    return tensor_to_vec(parafac2_to_tensor(parafac2_tensor))
