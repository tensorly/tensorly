from typing import List

import numpy as np
from sklearn.preprocessing import normalize

# Authors: Hratch Baghdassarian <hmbaghdassarian@gmail.com>, Erick Armingol <earmingol14@gmail.com>
# similarity metrics for tensor decompsoitions

def stack_loading_matrices(factors: List[np.ndarray]) -> np.ndarray:
    """Vertically stack the loading/factor matrices.

    Parameters
    ----------
    factors : List[np.ndarray]
        The loading/factor matrices [A^1 ... A^R] for a decomposed low-rank tensor from its factors.

    Returns
    -------
    np.ndarray
        A vertically stacked matrix of shape [sum(tensor.shape)xR)] 
    """
    
    if not np.all([A.shape[1] for A in factors]):
        raise ValueError('factors should be a list of loading matrices')
    return np.vstack(factors)

def calc_corrindex(C: np.ndarray) -> float:
    """Calculate the CorrIndex.

    Parameters
    ----------
    C : np.ndarray
        absolute value of the product of two MxN matrices that have had their columns L2 normalized, 
        taking the complex conjugate transpose of the first matrix -- C = |X1^H * X2|

    Returns
    -------
    float
        CorrIndex metric [0,1]; lower score means higher between matrices
    """
    
    MN = C.shape[1]*C.shape[0]
    return (1/(MN))*(np.sum(np.abs(np.amax(C, 1) - 1)) + np.sum(np.abs(np.amax(C, 0) - 1)))

def corrindex(factors_1: List[np.ndarray], factors_2: List[np.ndarray], tol: float = 5e-16) -> float:
    """CorrIndex implementation to assess tensor decomposition outputs. 
    From Sobhani et al 2022 (https://doi.org/10.1016/j.sigpro.2022.108457). 
    Metric is scaling and column-permutation invariant.  

    Parameters
    ----------
    factors_1 : List[np.ndarray]
        The loading/factor matrices [A^1 ... A^R] for a low-rank tensor from its factors, output from first decomposition
    factors_2 : List[np.ndarray]
        The loading/factor matrices [A^1 ... A^R] for a low-rank tensor from its factors, output from second decomposition
    tol : float, optional
        precision threshold below to call the CorrIndex score 0, by default 1e-16
        
    Returns
    -------
    score : float
         CorrIndex metric [0,1]; lower score means higher similarity between matrices
    """   
    # get stacked loading matrices
    X_1 = stack_loading_matrices(factors_1)
    X_2 = stack_loading_matrices(factors_2)
    
    if X_1.shape != X_2.shape: 
        raise ValueError('Factor matrices should be of the same shapes')
    
    # normalize columns to L2 norm - even if ran decomposition with normalize_factors = True  
#   X/np.apply_along_axis(np.linalg.norm, 0, X) 
    X_1 = normalize(X_1, norm = 'l2', axis = 0)
    X_2 = normalize(X_2, norm = 'l2', axis = 0)
    
    C = np.abs(np.matmul(np.matrix(X_1).H, X_2))
    
    score = calc_corrindex(C)
    if score < tol:
        score = 0
    return score
