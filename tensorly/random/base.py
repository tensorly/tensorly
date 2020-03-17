import numpy as np
from numpy.linalg import qr
from ..kruskal_tensor import (kruskal_to_tensor, KruskalTensor,
                              kruskal_normalise)
from ..tucker_tensor import tucker_to_tensor
from ..mps_tensor import mps_to_tensor
from .. import backend as T
import warnings

def cp_tensor(*args, **kwargs):
    message = "'cp_tensor' is depreciated, please use 'random_kruskal' instead"
    warnings.warn(message, DeprecationWarning)
    return random_kruskal(*args, **kwargs)

def tucker_tensor(*args, **kwargs):
    message = "'tucker_tensor' is depreciated, please use 'tucker_tensor' instead"
    warnings.warn(message, DeprecationWarning)
    return random_tucker(*args, **kwargs)

def check_random_state(seed):
    """Returns a valid RandomState

    Parameters
    ----------
    seed : None or instance of int or np.random.RandomState(), default is None

    Returns
    -------
    Valid instance np.random.RandomState

    Notes
    -----
    Inspired by the scikit-learn eponymous function
    """
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)

    elif isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('Seed should be None, int or np.random.RandomState')

def random_kruskal(shape, rank, full=False, orthogonal=False, 
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
    random_kruskal : ND-array or 2D-array list
        ND-array : full tensor if `full` is True
        2D-array list : list of factors otherwise
    """
    if (rank > min(shape)) and orthogonal:
        warnings.warn('Can only construct orthogonal tensors when rank <= min(shape) but got '
                      'a tensor with min(shape)={} < rank={}'.format(min(shape), rank))

    rns = check_random_state(random_state)
    factors = [T.tensor(rns.random_sample((s, rank)), **context) for s in shape]
    weights = T.ones(rank, **context)
    if orthogonal:
        factors = [T.qr(factor)[0] for factor in factors]

    if full:
        return kruskal_to_tensor((weights, factors))
    elif normalise_factors:
        return kruskal_normalise((weights, factors))
    else:
        return KruskalTensor((weights, factors))

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
    rns = check_random_state(random_state)

    if isinstance(rank, int):
        rank = [rank for _ in shape]

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
        return core, factors

def random_mps(shape, rank, full=False, random_state=None, **context):
    """Generates a random MPS/ttrain tensor

    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : int
        rank of the MPS decomposition
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
    MPS_tensor : ND-array or 3D-array list
        * ND-array : full tensor if `full` is True
        * 3D-array list : list of factors otherwise
    """
    n_dim = len(shape) 

    if isinstance(rank, int):
        rank = [1] + [rank] * (n_dim-1) + [1]
    elif n_dim+1 != len(rank):
        message = 'Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {} while tl.ndim(tensor) + 1  = {}'.format(
            len(rank), n_dim + 1)
        raise(ValueError(message))

    # Make sure it's not a tuple but a list
    rank = list(rank)

    # Initialization
    if rank[0] != 1:
        message = 'Provided rank[0] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[0] to 1.'.format(rank[0])
        raise ValueError(message)
    if rank[-1] != 1:
        message = 'Provided rank[-1] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[-1] to 1.'.format(rank[0])
        raise ValueError(message)

    rns = check_random_state(random_state)
    factors = [T.tensor(rns.random_sample((rank[i], s, rank[i+1])), **context)\
               for i, s in enumerate(shape)]

    if full:
        return mps_to_tensor(factors)
    else:
        return factors
