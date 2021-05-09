import numpy as np
import warnings
import math
import sys

import tensorly as tl
from ...decomposition._base_decomposition import DecompositionMixin
from ...cp_tensor import (cp_to_tensor, CPTensor,
                          unfolding_dot_khatri_rao, cp_norm,
                          cp_normalize, validate_cp_rank)
from ...decomposition._cp import initialize_cp


def gcp(X, R, type='normal', opt='lbfgsb', mask=None, n_iter_max=1000, \
        init='random', printitn=10, state=None, factr=1e7, pgtol=1e-4, \
        fsamp=None, gsamp=None, oversample=1.1, sampler=None, \
        fsampler=None, rate=1e-3, decay=0.1, maxfails=1, epciters=1000, \
        festtol=-math.inf, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Generalized CANDECOMP/PARAFAC (GCP) decomposition via all-at-once optimization (OPT) [1]
    Computes a rank-'R' decomposition of 'tensor' such that::

      tensor = [|weights; factors[0], ..., factors[-1] |].

    GCP-OPT allows the use of a variety of statistically motivated loss functions
    suited to the data held in a tensor (i.e. continuous, discrete, binary, etc)

    Parameters
    ----------
    tensor : ndarray
    rank : int
      Number of components.
    type : str,
      Loss function, 'normal' by default
      User specified loss function, options include:
          'binary'    - Bernoulli distribution for binary data
          'count'     - Poisoon distribution for count data
          'normal'    - Gaussian distribution
          'huber (0.25) - Similar to Gaussian, robust to outliers
          'rayleigh'  - Rayleigh distribution for non-negative data
          'gamma'     - Gamma distribution for non-negative data
    opt : str
      Optimization method
        'lbfgsb'    - Bound-constrained limited-memory BFGS
        'sgd'       - Stochastic gradient descent (SGD)
        'adam'      - Momentum-based SGD method
        If 'tensor' is dense, all 3 options can be used, 'lbfgsb' by default.
        If 'tensor' is sparse, only 'sgd' and 'adam' can be used, 'adam' by
        default.
        Each method has specific parameters, see documentation
    mask : ndarray
      Specifies a mask, 0's for missing entries, 1's elsewhere, with
      the same shape as 'tensor'.
    n_iter_max : int
      Maximum number of outer iterations, 1000 by default.
    init : {random, svd}
      Initialization for factor matrices, 'random' by default.
    printitn : int
      Print every n iterations; 0 for no printing, 10 by default.
    random_state : {None, int, np.random.RandomState}
      Seed for reproducable random number generation
    factr : float
      Tolerance on the change of objective values (L-BFGS-B parameter). Defaults
      to 1e7.
    pgtol : float
      Projected gradient tolerance (L-BFGS-B parameter).  Defaults to 1e-5
    sampler : {uniform, stratified, semi-stratified}
      Type of sampling to use for stochastic gradient (SGD/ADAM/ADAGRAD parameter).
      Defaults to 'uniform'.
    gsamp : int
      Number of samples for stochastic gradient (SGD/ADAM/ADAGRAD parameter).
      Generally set to be O(sum(sz)*r). For stratified or semi-stratified, this
      may be two numbers: the number of nnz samples and the number of zero samples.
      If only one number is specified, then this value is used for both nnzs and
      zeros (total number of samples is 2x specified value in this case).
    fsampler : {uniform, stratified, custom}
      Type of sampling for estimating function value (SGD/ADAM/ADAGRAD parameter).
      Custom function handle is primarily useful in reusing the same samled
      elements across different tests.
    fsamp : int
      Number of samples to estimate funciton (SGD/ADAM/ADAGRAD parameter). This
      should generally be somewhat large since we want this sample to generate a
      reliable estimate of the true function value.
    oversample : float
      Factor to oversample when implicitly sampling zeros in the sparse case
      (SGD/ADAM/ADAGRAD parameter).  Defaults to 1.1. Only adjust for very small
      tensors.
    rate : float
      Initial learning rate (SGD/ADAM/ADAGRAD parameter). Defaults to 1e-3.
    decay : float
      Amount to decrease learning rate when progress stagnates, i.e. no change in
      objective function between epochs.  Defaults to 0.1.
    maxfails : int
      Number of times to decrease the learning rate (SGD/ADAM/ADAGRAD parameter).
      Defaults to 1, may be set to zero.
    epciters : int
      Iterations per epoch (SGD/ADAM/ADAGRAD parameter). Defaults to 1000.
    festtol : float
      Quit estimation of function if it goes below this level (SGD/ADAM/ADAGRAD parameter).
      Defaults to -inf.




    Returns
    -------

    Reference
    ---------
    [1] D. Hong, T. G. Kolda, J. A. Duersch, Generalized Canonical
        Polyadic Tensor Decomposition, SIAM Review, 62:133-163, 2020,
        https://doi.org/10.1137/18M1203626
    [2] T. G. Kolda, D. Hong, Stochastic Gradients for Large-Scale Tensor
        Decomposition. SIAM J. Mathematics of Data Science, 2:1066-1095,
        2020, https://doi.org/10.1137/19m1266265

    """
    # Timers, @@@@@ TODO

    # Initial setup
    tensor_order = tl.ndim(X)
    tensor_shape = tl.shape(X)
    tensor_size = 1
    for i in range(tensor_order):
        tensor_size *= tensor_shape[i]

    # Random set-up
    if state is not None:
        rng = tl.check_random_state(state)

    # @@@@@ Do I need an equivalent to the 'info' structure in Tensor Toolbox
    # code??  Captures params, tensor info details

    # initialize CP-tensor
    M = initialize_cp(X, R, init=init, random_state=rng)

    # Set up function, gradient, and bounds
    if type:
        fh, gh, lb = validate_type(type)


def validate_type(type):
    """ Validate 'type' is among the implemented loss/gradient/lower bound
    return loss function, gradient, and lower bound if available
    fail gracefully otherwise

    Parameters
    ----------
    type : string
      Type of loss function to use in the decompostion

    Returns
    -------
    fh : function handle
      Lamda function for specified loss
    gh : function handle
      Lamda function for the gradient of the loss
    lb : {0, -inf}
      Lower bound

    """
    if type == "normal" or type == 'gaussian':
        fh = lambda x, m: (x - m) ** 2
        gh = lambda x, m: 2 * (x - m)
        lb = -math.inf
    elif type == 'binary' or type == 'bernoulli-odds':
        fh = lambda x, m: math.log(m + 1) - x * math.log(m + 1e-10)
        gh = lambda x, m: 1 / (m + 1) - x / (m + 1e-10)
        lb = 0
    elif type == 'bernoulli-logit':
        fh = lambda x, m: math.log(math.exp(m) + 1) - x * m
        gh = lambda x, m: math.exp(m) / (math.exp(m) + 1) - x
        lb = -math.inf
    elif type == 'count' or type == 'poisson':
        fh = lambda x, m: m - x * math.log(m + 1e-10)
        gh = lambda x, m: 1 - x / (m + 1e-10)
        lb = 0
    elif type == 'poisson-log':
        fh = lambda x, m: math.exp(m) - x * m
        gh = lambda x, m: 1 - x / (m + 1e-10)
        lb = 0
    elif type == 'rayleigh':
        fh = lambda x, m: 2 * math.log(m + 1e-10) + (math.pi / 4) * ((x / (m + 1e-10)) ** 2)
        gh = lambda x, m: 2 / (m + 1e-10) - (math.pi / 2) * (x ** 2) / ((m + 1e-10) ** 3)
        lb = 0
    else:
        print("Type unsupported!!")
        sys.exit(1)

        return fh, gh, lb


def validate_opt(opt):
    """Validate 'opt' method is supported

    Parameters
    ----------
    opt : {lbfgsb, sgd, adam, adagrad}
        Optimization method

    Returns
    -------
    status : {1,0}
        Returns 1 if 'opt' not supported or implemented

    """
    status = 0
    if opt == "lbfgsb":
        print("Optimization: " + opt)
    elif opt == "sgd":
        print("Optimization: " + opt + " not yet implemented.")
        status = 1
    elif opt == "adam":
        print("Optimization: " + opt + " not yet implemented.")
        status = 1
    elif opt == "adagrad":
        print("Optimization: " + opt + " not yet implemented.")
        status = 1
    else:
        print("Optimization method not supported. Choose from: lbfgsb, sgd, ada, adagrad")
        status = 1

    return status

