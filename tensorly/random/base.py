import numpy as np
from numpy.linalg import qr
from ..kruskal_tensor import kruskal_to_tensor
from ..tucker_tensor import tucker_to_tensor
from .. import backend as T


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

def cp_tensor(shape, rank, full=False, orthogonal=False, random_state=None, **context):
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
    cp_tensor : ND-array or 2D-array list
        ND-array : full tensor if `full` is True
        2D-array list : list of factors otherwise
    """
    if (rank > min(shape)) and orthogonal:
        raise ValueError('Can only construct orthogonal tensors when rank <= '
                         'min(shape)')

    rns = check_random_state(random_state)
    factors = [T.tensor(rns.random_sample((s, rank)), **context) for s in shape]
    if orthogonal:
        factors = [T.qr(factor)[0] for factor in factors]

    if full:
        return kruskal_to_tensor(factors)
    else:
        return factors

def tucker_tensor(shape, rank, full=False, orthogonal=False, random_state=None, **context):
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

    for i, (s, r) in enumerate(zip(shape, rank)):
        if r > s:
            raise ValueError('The rank should be smaller than the tensor size, yet rank[{0}]={1} > shape[{0}]={2}.'.format(i, r, s))

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
        return tucker_to_tensor(core, factors)
    else:
        return core, factors
