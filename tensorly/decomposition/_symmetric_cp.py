import tensorly as tl
from ._base_decomposition import DecompositionMixin
from tensorly.tenalg import outer
import numpy as np
from ..cp_tensor import validate_cp_rank


def symmetric_power_iteration(tensor, n_repeat=10, n_iteration=10, verbose=False):
    """A single Robust Symmetric Tensor Power Iteration

    Parameters
    ----------
    tensor : tl.tensor
        input tensor to decompose, must be symmetric of shape (size, )*order
    n_repeat : int, default is 10
        number of initializations to be tried
    n_iterations : int, default is 10
        number of power iterations
    verbose : bool
        level of verbosity

    Returns
    -------
    (eigenval, best_factor, deflated)

    eigenval : float
        the obtained eigenvalue
    best_factor : tl.tensor
        the best estimated eigenvector
    deflated : tl.tensor of same shape as `tensor`
        the deflated tensor (i.e. without the estimated component)
    """
    order = tl.ndim(tensor)
    size = tl.shape(tensor)[0]

    if not tl.shape(tensor) == (size, )*order:
        raise ValueError('The input tensor does not have the same size along each mode.')

    # A list of candidates for each mode
    scores = []
    modes = list(range(1, order))

    for i in range(n_repeat):
        factor = tl.tensor(np.random.random_sample(size), **tl.context(tensor))

        for _ in range(n_iteration):
            for _ in range(order):
                factor = tl.tenalg.multi_mode_dot(tensor, [factor]*(order-1), modes=modes)
                factor = factor / tl.norm(factor, 2)

        score = tl.tenalg.multi_mode_dot(tensor, [factor]*order)
        scores.append(score) #round(score, 2))
        
        if (i == 0) or (score > best_score):
            best_score = score
            best_factor = factor

    if verbose:
        print(f'Best score of {n_repeat}: {best_score}')
    
    # Refine the init
    for _ in range(n_iteration):
        for _ in range(order):
            best_factor = tl.tenalg.multi_mode_dot(tensor, [best_factor]*(order-1), modes=modes)
            best_factor = best_factor / tl.norm(best_factor, 2)

    eigenval = tl.tenalg.multi_mode_dot(tensor, [best_factor]*order)
    deflated = tensor - outer([best_factor]*order)*eigenval

    if verbose:
        explained = tl.norm(deflated)/tl.norm(tensor)
        print(f'Eigenvalue: {eigenval}, explained: {explained}')

    return eigenval, best_factor, deflated


def symmetric_parafac_power_iteration(tensor, rank, n_repeat=10, n_iteration=10, verbose=False):
    """Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration

    Parameters
    ----------
    tensor : tl.tensor
        input tensor to decompose, must be symmetric of shape (size, )*order
    rank : int
        rank of the decomposition (number of rank-1 components)
    n_repeat : int, default is 10
        number of initializations to be tried
    n_iterations : int, default is 10
        number of power iterations
    verbose : bool
        level of verbosity

    Returns
    -------
    (weights, factor)

    weights : 1-D tl.tensor of length `rank`
        contains the eigenvalue of each eigenvector
    factor : 2-D tl.tensor of shape (size, rank)
        each column corresponds to one eigenvector
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)

    order = tl.ndim(tensor)
    size = tl.shape(tensor)[0]

    if not tl.shape(tensor) == (size, )*order:
        raise ValueError('The input tensor does not have the same size along each mode.')

    factor = []
    weights = []

    for _ in range(rank):
        eigenval, eigenvec, deflated = symmetric_power_iteration(tensor, n_repeat=n_repeat, n_iteration=n_iteration, verbose=verbose)
        factor.append(eigenvec)
        weights.append(eigenval)
        tensor = deflated

    factor = tl.stack(factor, axis=1)
    weights = tl.stack(weights)

    return weights, factor

class SymmetricCP(DecompositionMixin):
    """Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration

    Parameters
    ----------
    rank : int
        rank of the decomposition (number of rank-1 components)
    n_repeat : int, default is 10
        number of initializations to be tried
    n_iterations : int, default is 10
        number of power iterations
    verbose : bool
        level of verbosity

    Returns
    -------
    (weights, factor)

    weights : 1-D tl.tensor of length `rank`
        contains the eigenvalue of each eigenvector
    factor : 2-D tl.tensor of shape (size, rank)
        each column corresponds to one eigenvector
    """
    def __init__(self, rank, n_repeat=10, n_iteration=10, verbose=False):
        self.rank = rank
        self.n_repeat = n_repeat
        self.n_iteration = n_iteration
        self.verbose = verbose

    def fit_transform(self, tensor):
        self.decomposition_ = symmetric_parafac_power_iteration(tensor, self.rank, n_repeat=self.n_repeat,
                                                                n_iteration=self.n_iteration, verbose=self.verbose)
        return self.decomposition_
