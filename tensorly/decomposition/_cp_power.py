import tensorly as tl
from ._base_decomposition import DecompositionMixin
from ..cp_tensor import validate_cp_rank
from tensorly.tenalg import outer
import numpy as np

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>

# License: BSD 3 clause

def power_iteration(tensor, n_repeat=10, n_iteration=10, verbose=False):
    """A single Robust Tensor Power Iteration

    Parameters
    ----------
    tensor : tl.tensor
        input tensor to decompose
    n_repeat : int, default is 10
        number of initializations to be tried
    n_iteration : int, default is 10
        number of power iterations
    verbose : bool
        level of verbosity

    Returns
    -------
    (eigenval, best_factor, deflated)

    eigenval : float
        the obtained eigenvalue
    best_factors : tl.tensor list
        the best estimated eigenvector, for each mode of the input tensor
    deflated : tl.tensor of same shape as `tensor`
        the deflated tensor (i.e. without the estimated component)
    """
    order = tl.ndim(tensor)
    
    # A list of candidates for each mode
    scores = []
    
    for i in range(n_repeat):
        factors = [tl.tensor(np.random.random_sample(s), **tl.context(tensor)) for s in tl.shape(tensor)]

        for _ in range(n_iteration):
            for mode in range(order):
                factor = tl.tenalg.multi_mode_dot(tensor, factors, skip=mode)
                factor = factor / tl.norm(factor, 2)
                factors[mode] = factor
                
        score = tl.tenalg.multi_mode_dot(tensor, factors)
        scores.append(score) #round(score, 2))
        
        if (i == 0) or (score > best_score):
            best_score = score
            best_factors = factors

    if verbose:
        print(f'Best score of {n_repeat}: {best_score}')
    
    # Refine the init
    for _ in range(n_iteration):
        for mode in range(order):
            factor = tl.tenalg.multi_mode_dot(tensor, best_factors, skip=mode)
            factor = factor / tl.norm(factor, 2)
            best_factors[mode] = factor

    eigenval = tl.tenalg.multi_mode_dot(tensor, best_factors)
    deflated = tensor - outer(best_factors)*eigenval
    
    if verbose:
        explained = tl.norm(deflated)/tl.norm(tensor)
        print(f'Eigenvalue: {eigenval}, explained: {explained}')

    return eigenval, best_factors, deflated


def parafac_power_iteration(tensor, rank, n_repeat=10, n_iteration=10, verbose=0):
    """CP Decomposition via Robust Tensor Power Iteration

    Parameters
    ----------
    tensor : tl.tensor
        input tensor to decompose
    rank : int
        rank of the decomposition (number of rank-1 components)
    n_repeat : int, default is 10
        number of initializations to be tried
    n_iteration : int, default is 10
        number of power iterations
    verbose : bool
        level of verbosity

    Returns
    -------
    (weights, factors)

    weights : 1-D tl.tensor of length `rank`
        contains the eigenvalue of each eigenvector
    factors : list of 2-D tl.tensor of shape (size, rank)
        Each column of each factor corresponds to one eigenvector
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)

    order = tl.ndim(tensor)
    factors = []
    weights = []

    for _ in range(rank):
        eigenval, eigenvec, deflated = power_iteration(tensor, n_repeat=n_repeat, n_iteration=n_iteration, verbose=verbose)
        factors.append(eigenvec)
        weights.append(eigenval)
        tensor = deflated

    factors = [tl.stack([f[i] for f in factors], axis=1) for i in range(order)]
    weights = tl.stack(weights)

    return weights, factors




class CPPower(DecompositionMixin):
    """CP Decomposition via Robust Tensor Power Iteration

    Parameters
    ----------
    tensor : tl.tensor
        input tensor to decompose
    rank : int
        rank of the decomposition (number of rank-1 components)
    n_repeat : int, default is 10
        number of initializations to be tried
    n_iteration : int, default is 10
        number of power iterations
    verbose : bool
        level of verbosity

    Returns
    -------
    (weights, factors)

    weights : 1-D tl.tensor of length `rank`
        contains the eigenvalue of each eigenvector
    factors : list of 2-D tl.tensor of shape (size, rank)
        Each column of each factor corresponds to one eigenvector
    """
    def __init__(self, rank, n_repeat=10, n_iteration=10, verbose=0):
        self.rank = rank
        self.n_repeat = n_repeat
        self.n_iteration = n_iteration
        self.verbose = verbose

    
    def fit_transform(self, tensor):
        """Decompose an input tensor

        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose

        Returns
        -------
        CPTensor
            decomposed tensor
        """
        cp_tensor = parafac_power_iteration(tensor, rank=self.rank,
                                                 n_repeat=self.n_repeat,
                                                 n_iteration=self.n_iteration,
                                                 verbose=self.verbose)
        self.decomposition_ = cp_tensor 
        return cp_tensor

    def __repr__(self):
        return f'Rank-{self.rank} CP decomposition via Robust Tensor Power Iteration.'
