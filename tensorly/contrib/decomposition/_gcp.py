import numpy as np
import warnings
import math
import sys
import inspect
from scipy.optimize import fmin_l_bfgs_b

import tensorly as tl
from ...decomposition._base_decomposition import DecompositionMixin
from ...cp_tensor import (cp_to_tensor, CPTensor,
                          unfolding_dot_khatri_rao, cp_norm,
                          cp_normalize, validate_cp_rank)
from ...decomposition._cp import initialize_cp


def gcp(X, R, type='normal', opt='lbfgsb', mask=None, maxiters=1000, \
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
    init : {random, svd, cptensor}
      Initialization for factor matrices, 'random' by default.
      Initial guess normalized to ensure weights are one.
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
    Mfin : CPTensor
        Canonical polyadic decomposition of input tensor X

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
    nd = tl.ndim(X)
    sz = tl.shape(X)
    tsz = X.size
    X_context = tl.context(X)
    vecsz = 0
    for i in range(nd):
        # tsz *= sz[i]
        vecsz += sz[i]
    vecsz *= R
    W = mask

    # Random set-up
    if state is not None:
        state = tl.check_random_state(state)

    # @@@@@  TODO: Do I need an equivalent to the 'info' structure in Tensor Toolbox
    # code??  Captures params, tensor info details

    # capture stats(nnzs, zeros, missing)
    nnonnzeros = 0
    X = tl.tensor_to_vec(X)
    for i in X:
        if i > 0: nnonnzeros += 1
    X = tl.reshape(X,sz)
    nzeros = tsz - nnonnzeros
    nmissing = 0
    if W is not None:
        W = tl.tensor_to_vec(W)
        for i in range(tl.shape(W)[0]):
            if W[i] > 0 : nmissing += 1
        W = tl.reshape(W,sz)

    # Set up function, gradient, and bounds
    # @@@@@@ fh, gh, lb = validate_type(type) # old-way, needs troubleshooting
    fh = None
    gh = None
    lb = None
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

    # initialize CP-tensor and make a copy to work with so as to have the starting guess
    M0 = initialize_cp(X, R, init=init, random_state=state)
    wghts0 = tl.copy(M0[0])
    fcts0 = []
    for i in range(nd):
        f = tl.copy(M0[1][i])
        fcts0.append(f)
    M = CPTensor((wghts0,fcts0))

    # check optimization method
    if validate_opt(opt):
        print("Choose optimization method from: {lbfgsb}")
        sys.exit(1)
    use_stoc = False
    if opt != 'lbfgsb':
        use_stoc = True

    # @@@@@ TODO: implement stochastic gradient optimization methods, this is were f and g sampling set-up will go

    # Welcome message
    if printitn > 0:
        print("GCP-OPT-{} (Generalized CP Tensor Decomposition)\n".format(opt))
        print("Tensor size: {} ({} total entries)".format(sz,tsz))
        if nmissing > 0:
            print("Missing entries: {} ({})".format(nmissing, 100*nmissing/tsz))
        print("Generalized function Type: {}".format(type))
        print("Objective function: {}".format(inspect.getsource(fh)))
        print("Gradient function: {}".format(inspect.getsource(gh)))
        print("Lower bound of factor matrices: {}".format(lb))
        print("Optimization method: {}".format(opt))
        if use_stoc:
            print("Max iterations (epochs): {}".format(maxiters))
            print("Iterations per epoch: {}".format(epciters))
            print("Learning rate / decay / maxfails: {} {} {}".format(rate, decay, maxfails))
            print("Function Sampler: ") # TODO sampling set up prepare string for this field
            print("Gradient Sampler: ")  # TODO sampling set up prepare string for this field
        else:
            print("Max iterations: {}".format(maxiters))
            print("Projected gradient tolerance: {}".format(pgtol))

    # Make like a zombie and start decomposing
    Mfin = None
    if opt=='lbfgsb':
        # set up bounds for l-bfgs-b if lb = 0
        bounds = None
        if lb == 0:
            lb = tl.zeros(tsz)
            ub = math.inf*tl.ones(tsz)
        fcn = lambda x: tt_gcp_fg(vec2factors(x, sz, R, X_context), X, fh, gh)
        m = factors2vec(M[1])
        x, f, info_dict = fmin_l_bfgs_b(fcn, m, approx_grad=False, bounds=None, \
                                        pgtol=pgtol, factr=factr, maxiter=maxiters)
        if printitn > 0:
            print("\nFinal objective: {}".format(f))
            print("Setup time: ") # @@@@@ TODO attached to setting up timers
            print("Main loop time: ") # @@@@@ TODO attached to setting up timers
            print("Outer iterations:")
            print("Total iterations: {}".format(info_dict['nit']))
            print("L-BFGS-B exit message: {} ({})".format(info_dict['task'], info_dict['warnflag']))
        Mfin = vec2factors(x, sz, R, X_context)

    if use_stoc:
        # TODO perform optimization with SGD/ADAM/ADAGRAD, yet to be implemented
        pass


def vec2factors(vec, shape, rank, context = None):
    """Wrapper function detailed in Appendix C [1]
    Builds a set of N matrices, where the k-th matrix is shape(k) x rank in dimension

    Parameters
    ----------
    vec : ndarray
        vector of values to proliferate matrices with
    shape: tensor shape
        shape of tensor dictates number of rows in each matrix
    rank: int
        number of columns in each matrix, *** rank cannot be > dimension of smallest mode ***

    Returns
    -------
    M1 : CPTensor
        CPTensor with factor matrices formed by 'vec'
    """
    numFacts = len(shape)
    factors = []
    place = 0
    for i in range(numFacts):
      factor = np.zeros((rank*shape[i]), **context)
      for j in range(shape[i]*rank):
        factor[j] = vec[j+place]
      factor = tl.tensor(factor.reshape((rank, shape[i])), **context)
      factors.append(tl.transpose(factor))
      place += shape[i]*rank
    M1 = CPTensor((tl.ones(rank,), factors))
    return M1


def factors2vec(factors):
    """Wrapper function detailed in Appendix C [1]
    Stacks the column vectors of a set of matrices into a single vecto

    Parameters
    ---------
    factors : list of ndarrays
        Factor matrices or Gradient wrt factor gradient

    Returns
    -------
    vec : ndarry
        column-wise vectorization of a list of matrices
    """
    vec = None
    for factor in factors:
        if vec is None:
            vec = tl.tensor_to_vec(tl.unfold(factor,1))
        else:
            vec = tl.concatenate([vec,tl.tensor_to_vec(tl.unfold(factor,1))])
    return vec

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
    fh = None
    gh = None
    lb = None
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

def tt_gcp_fg(M, X, f, g, W = None, computeF = True, computeG = True, vectorG = True):
    """Loss function and gradient for generalized CP decomposition.

    Parameters
    ----------
    M : CPTensor
    X : ndarray
        Dense tensor
    f : Function handle
        elementwise loss of the form f(x,m)
    g : Function handle
        Elementwise intermediate gradient of the form g(x,m)
    W : ndarray
        Weight tensor, 1's for known values, 0's for missing values.  Function/gradient is only computed w.r.t
        know values. Setting W =[] indicates no missing data.
    computeF : boolean
        Include computation of the loss function. Default is true.
    computeG : boolean
        Include computation of the gradient.
    vectorG : boolean
        Reshape gradient matrices into a single vector.

    Returns
    -------
    F : scalar
        Loss function value
    G : ndarray(s)
        If vectorG = False, G is a list of matrices where G[k] is the same size as the k-th factor matrix
        Otherwise, G is the gradient in vector form
    """
    # setup
    Mfull = M.to_tensor()
    Mv = tl.tensor_to_vec(Mfull)
    Xv = tl.tensor_to_vec(X)

    F = None
    G = []
    # calculate loss
    if computeF:
        Fvec = f(Xv,Mv)
        if W is not None:
            # TODO handle applying weight tensor, probably need to vec it then elementwise product
            pass
        F = Fvec.sum()
    # calculate gradient
    if computeG:

        Y = g(Xv,Mv)
        Y = tl.reshape(Y, tl.shape(X))

        if W is not None:
            # TODO handle applying weight tensor, probably need to vec it then elementwise product
            pass

        for i in range(len(M[1])):
            mGrad = tl.unfolding_dot_khatri_rao(Y, M, i)
            G.append(mGrad)
        if vectorG:
            G = factors2vec(G)

    return F, G
