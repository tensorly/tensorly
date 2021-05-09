import numpy as np
import warnings
import math

import tensorly as tl
from ...decomposition._base_decomposition import DecompositionMixin
from ...random import check_random_state, random_cp
from ...base import unfold
from ...cp_tensor import (cp_to_tensor, CPTensor,
                          unfolding_dot_khatri_rao, cp_norm,
                          cp_normalize, validate_cp_rank)
from ...decomposition._cp import initialize_cp as init_cp_tensor

def gcp(tensor, rank, type='normal', opt='lbfgsb', mask=None,
        n_iter_max=1000, init='random', printitn=10, random_state=None):
    """Generalized CANDECOMP/PARAFAC (GCP) decomposition via optimization
    Computes a rank-'rank' decomposition of 'tensor' [1]_ such that ::

        tensor = [|weights; factors[0], ..., factors[-1] |].

    Parameters
    ----------
    tensor : ndarray
    rank : int
        Number of components.
    type : str, optimization loss function, 'normal' by default
        User specified loss function, options include:
            'binary'    - Bernoulli distribution for binary data
            'count'     - Poisoon distribution for count data
            'normal'    - Gaussian distribution
            'huber (0.25) - Similar to Gaussian, robust to outliers
            'rayleigh'  - Rayleigh distribution for non-negative data
            'gamma'     - Gamma distribution for non-negative data
    opt : str, optimization method
        'lbfgsb'    - Bound-constrained limited-memory BFGS
        'sgd'       - Stochastic gradient descent (SGD)
        'adam'      - Momentum-based SGD method
        If 'tensor' is dense, all 3 options can be used, 'lbfgsb' by default.
        If 'tensor' is sparse, only 'sgd' and 'adam' can be used, 'adam' by
        default.
        Each method has specific parameters, see documentation
    mask : ndarray
        specifies a mask, 0's for missing entries, 1's elsewhere, with
        the same shape as 'tensor'.
    n_iter_max : int
        Maximum number of outer iterations, 1000 by default.
    init : str
        Initialization for factor matrices, 'random' by default.
    printitn : int
        Print every n iterations; 0 for no printing, 10 by default.
    random_state : {None, int, np.random.RandomState}

    References
    ----------
    * D. Hong, T. G. Kolda, J. A. Duersch, Generalized Canonical
      Polyadic Tensor Decomposition, SIAM Review, 62:133-163, 2020,
      https://doi.org/10.1137/18M1203626
    * T. G. Kolda, D. Hong, Stochastic Gradients for Large-Scale Tensor
      Decomposition. SIAM J. Mathematics of Data Science, 2:1066-1095,
      2020, https://doi.org/10.1137/19m1266265
    """
    # Timers, @@@@@ TODO

    # Initial setup
    tensor_order = tl.ndim(tensor)
    tensor_shape = tl.shape(tensor)
    tensor_size = tensor.size       # @@@@@ TODO fix for backend compatability

    # Random set-up
    rng = check_random_state(random_state)

    # @@@@@ Do I need an equivalent to the 'info' structure in Tensor Toolbox
    # code??  Captures params, tensor info details

    # initialize CP-tensor
    cpTensor = init_cp_tensor(tensor, rank, init,random_state)

    # Set up function, gradient, and bounds
    if type:
        fh, gh, lb = check_type(type)


def check_type(type):
    """ check 'type' is among the implemented loss/gradient/lower bound
    return loss function, gradient, and lower bound if available
    fail gracefully otherwise
    """
    if type == "normal" or type == 'gaussian':
        fh = lambda x,m: (x-m)**2
        gh = lambda x,m: 2*(x-m)
        lb = -math.inf
    elif type == 'binary' or type == 'bernoulli-odds':
        fh = lambda x,m: math.log(m+1) - x*math.log(m + 1e-10)
        gh = lambda x,m: 1/(m+1) - x/(m + 1e-10)
        lb = 0
    elif type == 'bernoulli-logit':
        fh = lambda x,m: math.log(math.exp(m)+1) - x*m
        gh = lambda x,m: math.exp(m)/(math.exp(m) + 1) - x
        lb = -math.inf
    elif type == 'count' or type == 'poisson':
        fh = lambda x,m: m - x*math.log(m + 1e-10)
        gh = lambda x,m: 1 - x/(m + 1e-10)
        lb = 0
    elif type == 'poisson-log':
        fh = lambda x,m: math.exp(m) - x*m
        gh = lambda x,m: 1 - x/(m + 1e-10)
        lb = 0
    elif type == 'rayleigh':
        fh = lambda x,m: 2*math.log(m + 1e-10) + (math.pi/4)*((x/(m + 1e-10))**2)
        gh = lambda x,m: 2/(m+1e-10) - (math.pi/2)*(x**2)/((m + 1e-10)**3)
        lb = 0
    else:
        print("spazz mode, returning normal to compensate")
        fh = lambda x, m: (x - m) ** 2
        gh = lambda x, m: 2 * (x - m)
        lb = -math.inf

    return fh, gh, lb



