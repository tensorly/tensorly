import numpy as np
from ..cp_tensor import (cp_to_tensor, CPTensor,
                         cp_normalize, validate_cp_rank)
from ..tucker_tensor import tucker_to_tensor, TuckerTensor, validate_tucker_rank
from ..tt_tensor import tt_to_tensor, TTTensor, validate_tt_rank
from ..tt_matrix import tt_matrix_to_tensor, TTMatrix, validate_tt_matrix_rank
from ..tr_tensor import TRTensor, tr_to_tensor, validate_tr_rank
from ..parafac2_tensor import parafac2_to_tensor, Parafac2Tensor
from .. import backend as T
from ..utils import DefineDeprecated
import warnings


def random_tensor(shape, random_state=None, **context):
    """Create a random tensor
    """
    rns = T.check_random_state(random_state)
    return T.tensor(rns.random_sample(shape), **context)


def random_parafac2(shapes, rank, full=False, random_state=None,
                    normalise_factors=True, **context):
    """Generate a random PARAFAC2 tensor

    Parameters
    ----------
    shape : tuple
        A tuple where each element represents the shape of a matrix
        represented by the PARAFAC2 model. The second element in each
        shape-tuple must be
        constant.
    rank : int or int list
        rank of the Parafac2 decomposition
    full : bool, optional, default is False
        if True, a full tensor is returned otherwise,
        the decomposed tensor is returned
    random_state : `np.random.RandomState`
    """
    rns = T.check_random_state(random_state)
    if not all(shape[1] == shapes[0][1] for shape in shapes):
        raise ValueError('All matrices must have equal number of columns.')
    
    projection_matrices = [
        T.qr(T.tensor(rns.random_sample((shape[0], rank)), **context))[0]
            for shape in shapes
    ]
    weights, factors = random_cp(
        [len(shapes), rank, shapes[0][1]], rank=rank, normalise_factors=False, 
        random_state=rns,  **context
    )

    parafac2_tensor = Parafac2Tensor((weights, factors, projection_matrices))

    if full:
        return parafac2_to_tensor(parafac2_tensor)
    else:
        return parafac2_tensor


def random_cp(shape, rank, full=False, orthogonal=False, 
                   random_state=None, normalise_factors=True, **context):
    """Generates a random CP tensor

    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : int
        rank of the CP decomposition
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    orthogonal : bool, optional, default is False
        if True, creates a tensor with orthogonal components
    random_state : `np.random.RandomState`
    context : dict
        context in which to create the tensor

    Returns
    -------
    random_cp : ND-array or 2D-array list
        ND-array : full tensor if `full` is True
        2D-array list : list of factors otherwise
    """
    rank = validate_cp_rank(shape, rank)
    if (rank > min(shape)) and orthogonal:
        warnings.warn('Can only construct orthogonal tensors when rank <= min(shape) but got '
                      'a tensor with min(shape)={} < rank={}'.format(min(shape), rank))

    rns = T.check_random_state(random_state)
    factors = [T.tensor(rns.random_sample((s, rank)), **context) for s in shape]
    weights = T.ones(rank, **context)
    if orthogonal:
        factors = [T.qr(factor)[0] for factor in factors]

    if full:
        return cp_to_tensor((weights, factors))
    elif normalise_factors:
        return cp_normalize((weights, factors))
    else:
        return CPTensor((weights, factors))

def random_tucker(shape, rank, full=False, orthogonal=False, random_state=None, **context):
    """Generates a random Tucker tensor

    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : int or int list
        rank of the Tucker decomposition
        if int, the same rank is used for each mode
        otherwise, dimension of each mode
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    orthogonal : bool, optional, default is False
        if True, creates a tensor with orthogonal components
    random_state : `np.random.RandomState`

    Returns
    -------
    tucker_tensor : ND-array or (ND-array, 2D-array list)
        ND-array : full tensor if `full` is True
        (ND-array, 2D-array list) : core tensor and list of factors otherwise
    """
    rns = T.check_random_state(random_state)

    rank = validate_tucker_rank(shape, rank)

    if orthogonal:
        for i, (s, r) in enumerate(zip(shape, rank)):
            if r > s:
                warnings.warn('Selected orthogonal=True, but selected a rank larger than the tensor size for mode {0}: '
                             'rank[{0}]={1} > shape[{0}]={2}.'.format(i, r, s))

    factors = []
    for (s, r) in zip(shape, rank):
        if orthogonal:
            factor = T.tensor(rns.random_sample((s, s)), **context)
            Q, _= T.qr(factor)
            factors.append(T.tensor(Q[:, :r]))
        else:
            factors.append(T.tensor(rns.random_sample((s, r)), **context))

    core = T.tensor(rns.random_sample(rank), **context)
    if full:
        return tucker_to_tensor((core, factors))
    else:
        return TuckerTensor((core, factors))


def random_tt(shape, rank, full=False, random_state=None, **context):
    """Generates a random TT/MPS tensor

    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : int
        rank of the TT decomposition
        must verify rank[0] == rank[-1] ==1 (boundary conditions)
        and len(rank) == len(shape)+1
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    random_state : `np.random.RandomState`
    context : dict
        context in which to create the tensor

    Returns
    -------
    TT_tensor : ND-array or 3D-array list
        * ND-array : full tensor if `full` is True
        * 3D-array list : list of factors otherwise
    """
    n_dim = len(shape) 

    rank = validate_tt_rank(shape, rank)

    # Make sure it's not a tuple but a list
    rank = list(rank)

    # Initialization
    if rank[0] != 1:
        message = 'Provided rank[0] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[0] to 1.'.format(rank[0])
        raise ValueError(message)
    if rank[-1] != 1:
        message = 'Provided rank[-1] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[-1] to 1.'.format(rank[0])
        raise ValueError(message)

    rns = T.check_random_state(random_state)
    factors = [T.tensor(rns.random_sample((rank[i], s, rank[i+1])), **context)\
               for i, s in enumerate(shape)]

    if full:
        return tt_to_tensor(factors)
    else:
        return TTTensor(factors)


def random_tt_matrix(shape, rank, full=False, random_state=None, **context):
    """Generates a random tensor in TT-Matrix format

    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : int
        rank of the TT decomposition
        must verify rank[0] == rank[-1] ==1 (boundary conditions)
        and len(rank) == len(shape)+1
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    random_state : `np.random.RandomState`
    context : dict
        context in which to create the tensor

    Returns
    -------
    TT_tensor : ND-array or 3D-array list
        * ND-array : full tensor if `full` is True
        * 3D-array list : list of factors otherwise
    """
    n_dim = len(shape) // 2
    left_shape = shape[:n_dim]
    right_shape = shape[n_dim:]

    rank = validate_tt_matrix_rank(shape, rank)

    factors = []
    for i in range(n_dim):
         factors.append(random_tensor((rank[i], left_shape[i], right_shape[i], rank[i + 1]),
                                             random_state=random_state, **context))

    if full:
        return tt_matrix_to_tensor(factors)
    else:
        return TTMatrix(factors)


def random_tr(shape, rank, full=False, random_state=None, **context):
    """Generates a random TR tensor

    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : List[int]
        rank of the TR decomposition
        must verify rank[0] == rank[-1] (boundary conditions)
        and len(rank) == len(shape)+1
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    random_state : `np.random.RandomState`
    context : dict
        context in which to create the tensor

    Returns
    -------
    TR_tensor : ND-array or 3D-array list
        * ND-array : full tensor if `full` is True
        * 3D-array list : list of factors otherwise
    """
    n_dim = len(shape)

    rank = validate_tr_rank(shape, rank)

    # Make sure it's not a tuple but a list
    rank = list(rank)

    # Initialization
    if rank[0] != rank[-1]:
        message = f'Provided rank[0] == {rank[0]} and rank[-1] == {rank[-1]} but boundary conditions dictatate rank[0] == rank[-1].'
        raise ValueError(message)

    rns = T.check_random_state(random_state)
    factors = [T.tensor(rns.random_sample((rank[i], s, rank[i + 1])), **context) for i, s in enumerate(shape)]

    if full:
        return tr_to_tensor(factors)
    else:
        return TRTensor(factors)


random_kruskal = DefineDeprecated(deprecated_name='random_kruskal', use_instead=random_cp)
random_mps = DefineDeprecated(deprecated_name='random_mps', use_instead=random_tt)
