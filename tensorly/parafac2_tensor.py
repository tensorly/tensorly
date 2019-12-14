"""Utility class and functions for the PARAFAC2 decomposition [1]_

These utility functions assume that the second mode ``(factors[1])`` evolve over
the first mode (factors[0]). Therefore, there are ``len(factors[1])`` separate 
factor matrices in the second mode. This decomposition is implemented in the same 
way as the direct fitting method described in [2] (except that the evolving mode 
and the mode that the tensor evolves over is changed). Mathematically this is 
equivalent to saying ``B[i] = P[i]@B`` for some matrix ``P[i].T@P[i] = I``, 
where ``I`` is an identity matrix.

References
----------
  .. [1] Kiers, H.A.L., ten Berge, J.M.F. and Bro, R. (1999), 
         PARAFAC2—Part I. A direct fitting algorithm for the PARAFAC2 model. 
         J. Chemometrics, 13: 275-294.
"""

# Authors: Marie Roald
#          Yngve Mardal Moe

from . import backend as T
import warnings
from collections.abc import Mapping


class Parafac2Tensor(Mapping):
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
    def from_kruskaltensor(self, kruskal_tensor, parafac2_tensor_ok=False):
        """Create a Parafac2Tensor from a KruskalTensor

        Parameters:
        -----------
        kruskal_tensor: KruskalTensor or Parafac2Tensor
            If it is a Parafac2Tensor, then the argument ``parafac2_tensor_ok`` must be True'
        parafac2_tensor: bool (optional)
            Whether or not Parafac2Tensors can be used as input.

        Returns:
        --------
        Parafac2Tensor
            Parafac2Tensor with factor matrices and weigths extracted from a KruskalTensor
        """
        if parafac2_tensor_ok and len(kruskal_tensor) == 3:
            return Parafac2Tensor(kruskal_tensor)
        elif len(kruskal_tensor) == 3:
            raise TypeError('Input is not a KruskalTensor. If it is a Parafac2Tensor, then the argument ``parafac2_tensor_ok`` must be True')
        
        weights, (A, B, C) = kruskal_tensor
        Q, R = T.qr(B)
        projections = [Q for _ in A]
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

        inner_product = projection.T@projection
        if T.norm(inner_product - T.eye(rank)) > 1e-10:
            raise ValueError(
                'All the projection matrices must be orthonormal, that is, P.T@P = I. '
                'However, T.norm(projection[{}].T@projection[{}] - T.eye(rank)) = {}'.format(
                    i, i, T.norm(inner_product - T.eye(rank)) 
                )
            )
        
        shape.append((current_mode_size, *[f.shape[0] for f in factors[2:]]))  # Tuple unpacking to possibly support higher order PARAFAC2 tensors in the future

    
    for i, factor in enumerate(factors[1:]):
        current_mode_size, current_rank = T.shape(factor)
        if current_rank != rank:
            raise ValueError('All the factors of a PARAFAC2 tensor should have the same number of columns.'
                             'However, factors[0].shape[1]={} but factors[{}].shape[1]={}.'.format(
                                 rank, i, T.shape(factor)[1]))

    if weights is not None and len(weights) != rank:
        raise ValueError('Given factors for a rank-{} PARAFAC2 tensor but len(weights)={}.'.format(
            rank, len(weights)))
        
    return tuple(shape), rank


def parafac2_normalise(parafac2_tensor, copy=False):
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
    _, rank, _ = _validate_parafac2_tensor(parafac2_tensor)
    weights, factors, projections = parafac2_tensor
    
    if (not copy) and (weights is None):
        warnings.warn('Provided copy=False and weights=None: a new Parafac2Tensor'
                      'with new weights and factors normalised inplace will be returned.')
        weights = T.ones(rank, **T.context(factors[0]))
    
    if copy:
        factors = [T.copy(f) for f in factors]
        projections = [T.copy(p) for p in projections]
        if weights is not None:
            factors[0] *= weights
        weights = T.ones(rank, **T.context(factors[0]))
    else:
        factors[0] *= weights
        weights = 0
        
    for factor in factors:
        scales = T.norm(factor, axis=0)
        weights *= scales
        scales_non_zero = T.where(scales==0, T.ones(T.shape(scales), **T.context(factors[0])), scales)
        factor /= scales_non_zero
        
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


def _get_projected_tensor_slice(tensor_slice, projection):
    """Get one slice of the projected tensor used for the PARAFAC step of a PARAFAC2 decompostion."""
    return T.dot(projection.T, tensor_slice)


def _get_projected_tensor(tensor, projections, out=None):
    """Get the projected tensor used for the PARAFAC step of a PARAFAC2 decompostion."""
    if out is None:
        I = len(tensor)
        J = projections[0].shape[1]
        K = tensor[0].shape[1]
        out = T.zeros((I, J, K), **T.context(tensor[0]))

    for i, (tensor_slice, projection) in enumerate(zip(tensor, projections)):
        out[i] = _get_projected_tensor_slice(tensor_slice, projection)

    return out


def parafac2_to_slice(parafac2_tensor, slice_idx, validate=True):
    """Generate a single slice along the first mode from the PARAFAC2 tensor.

    Generates a single slice along the first mode from a PARAFAC2 decomposition.
    That is, ``X[slice_idx] = P[slice_idx] B diag(A[slice_idx]) C^T``.

    Parameters
    ---------
    parafac2_tensor : Parafac2Tensor
        The decomposition that we wish to construct a slice from
    slice_idx : int
        The index of the slice we wish to construct
    
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
    """Generate all slices along the first mode from a PARAFAC2 tensor.

    Generates a list of all slices from a PARAFAC2 tensor. A list is returned
    since the tensor might have varying size along the second mode. To return
    a tensor, see the ``parafac2_to_tensor`` function instead.shape

    Parameters
    ---------
    parafac2_tensor : Parafac2Tensor
        The decomposition that we wish to construct slices from.
    
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
    """Construct a full tensor from a PARAFAC2 decomposition.

    Constructs the full tensor from the PARAFAC2 decomposition.
    The second mode might have different shapes, so the tensor will have shape
    ``[A.shape[1], max_P_len, C.shape[1]]``, where `max_P_len`` is the maximum
    size of the projection matrices of the decomposition. Uneven slices are
    padded with zero to construct a tensor.

    Parameters
    ----------
    parafac2_tensor : Parafac2Tensor
        The PARAFAC2 decomposition to construct the tensor from.
    
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
        tensor[i, :length] = slice_
    return tensor
