import tensorly as tl
# from ..random import check_random_state
# rng = check_random_state(1)

import time
import random
import numpy as np


from TensorToolbox.core import TensorWrapper, TTvec, TTmat

LINEAR_INTERPOLATION = 'LinearInterpolation'
LAGRANGE_INTERPOLATION = 'LagrangeInterpolation'
AVAIL_SURROGATE_TYPES = [LINEAR_INTERPOLATION, LAGRANGE_INTERPOLATION]



class STT():
    """ Constructor of the Spectral Tensor Train approximation :cite:`Bigoni2015`. Given a function ``f(x,theta,params):(Is, It) -> R``
    with ``dim(Is)=n`` and ``dim(It)=d``, construct an approximation of ``g(theta,params): It -> h_t(Is)``. For example ``Is`` could be the discretization of a spatial dimension, and ``It`` some parameter space, so that ``f(x,theta,params)`` describes a scalar field depending some parameters that vary in ``It``. The ``params`` in the definition of ``f`` can be constants used by the function or other objects that must be passed to the function definition.

    :param function f: multidimensional function to be approximated with format ``f(x,theta,params)``
    :param list grids: this is a list with ``len(grids)=dim(Is)+dim(It)`` which can contain:
        1-dimensional numpy.array of points discretizing the i-th dimension,
    :param object params: any list of parameters to be passed to the function ``f``
    :param int range_dim: define the dimension of the spatial dimension ``Is``. For functionals ``f(theta,params)``, ``dim(Is)=0``. For scalar fileds in 3D, ``dim(Is)=3``.
    :param strung ftype: 'serial' if it can only evaluate the function pointwise,
       'vector' if it can evaluate many points at once.
    :param str surrogate_type: whether the surrogate will be an interpolating surrogate (``TensorTrain.LINEAR_INTERPOLATION`` or ``TensorTrain.LAGRANGE_INTERPOLATION``)
    :param bool empty: Creates an instance without initializing it.

    .. note:: For a description of the remaining parameters see :py:class:`TTvec`.
    .. document private functions
    .. automethod:: __call__

    """

    def __init__(self, f, grids, params, range_dim=0,
                surrogate_type=None,
                 eps=1e-4, method="svd",
                 rs=None, delta=1e-4, maxit=100,
                 mv_eps=1e-5, mv_maxit=100,
                 ):


        ##########################################################
        # List of attributes
        #
        self.generic_approx = None  # np.array of TT approximation (can be TTvec or QTTvec)
        self.TTapprox = None  # np.array of TTvec approximations
        self.Xs_space = None  # List of points (defining the space grid)
        self.space_shape = None  # Shape of the space dimension
        self.Xs_params = None  # np.array of List of points (defining the parameter grid)

        self.interpolation_type = None  # Can be Projection or Interpolation
        self.barycentric_weights = None  # np.array of List of barycentric weights for Lagrange Interp

        self.range_dim = None  # Number of dimensions of the spatial space
        self.param_dim = None  # Number of dimensions of the parameter space
        self.TW = None  # Tensor wrapper containing caching the function evaluations
        self.params = None  # Parameters to be passed to f

        # Parameters to be reset on restarting
        self.f = None  # Function to be approximated


        # End list of attributes
        #########################################################

        self.f = f
        self.params = params
        self.interpolation_type = surrogate_type

        # Store all the tt approximation parameters
        self.method = method
        self.eps = eps
        self.rs = rs
        self.delta = delta
        self.maxit = maxit
        self.mv_eps = mv_eps
        self.mv_maxit = mv_maxit

        self.range_dim = range_dim
        self.param_dim = len(grids) - self.range_dim
        if self.param_dim < 0: raise AttributeError("The grids argument must respect len(grids) >= range_dim")

        # Store grid for spatial space
        #########################################################
        self.Xs_space = []
        for i in range(self.range_dim):
            if isinstance(grids[i], np.ndarray):
                self.Xs_space.append(grids[i])
            else:
                raise AttributeError("The grids argument must contain np.ndarray in the elements grids[:range_dim]")
        if self.range_dim == 0:
            self.Xs_space.append(np.array([0]))

        # Set the shape of the space
        self.space_shape = tuple([len(x) for x in self.Xs_space])

        # Initialize variables
        self.Xs_params = np.empty(self.space_shape, dtype=list)

        # Construct grids and weights if needed for parameter space
        for point, val in np.ndenumerate(self.Xs_params):
            self.Xs_params[point] = [None] * self.param_dim

            for i, grid in enumerate(grids[self.range_dim:]):
                if isinstance(grid, np.ndarray):
                    self.Xs_params[point][i] = grid
                else:
                    raise AttributeError("The %d argument of grid is none of the types accepted." % i)

        # Definition of the Tensor Wrapper (works for homogeneous grid on param space)
        #########################################################
        self.TW = TensorWrapper(self.f,
                                self.Xs_params[tuple([0] * max(self.range_dim, 1))],
                                params=self.params
                                )

        # Build
        #########################################################

        self.generic_approx = np.empty(self.space_shape, dtype=TTvec)
        self.generic_approx[(0,)] = TTvec(self.TW)
        self.generic_approx[(0,)].build(eps=self.eps, method=self.method,
                                         rs=self.rs,
                                         delta=self.delta, maxit=self.maxit,
                                         mv_eps=self.mv_eps,
                                         mv_maxit=self.mv_maxit
                                         )

        # Prepares the TTapprox from the generic_approx
        if self.TTapprox is None:
            self.TTapprox = np.empty(self.space_shape, dtype=TTvec)

        for (point, gen_app) in np.ndenumerate(self.generic_approx):
            self.TTapprox[point] = gen_app

        # Prepares the surrogate
        if self.interpolation_type == LAGRANGE_INTERPOLATION:
            if self.barycentric_weights is None:
                self.barycentric_weights = np.empty(self.space_shape, dtype=list)

            for (point, _), (_, X) in \
                    zip(np.ndenumerate(self.barycentric_weights), np.ndenumerate(self.Xs_params)):
                self.barycentric_weights[point] = [BarycentricWeights(X[i]) for i in range(self.param_dim)]


    def __call__(self, x_in, verbose=False):
        """ Evaluate the surrogate on points ``x_in``

        :param np.ndarray x_in: 1 or 2 dimensional array of points in the parameter space where to evaluate the function. In 2 dimensions, each row is an entry, i.e. ``x_in.shape[1] == self.param_dim``

        :return: an array with dimension equal to the space dimension (``range_dim``) plus one. If ``A`` is the returned vector and ``range_dim=2``, then ``A[i,:,:]`` is the value of the surrogate for ``x_in[i,:]``

        """

        if not isinstance(x_in, np.ndarray) or x_in.ndim not in [1, 2]: raise AttributeError(
            "The input variable must be a 1 or 2 dimensional numpy.ndarray")

        orig_ndim = x_in.ndim
        x = x_in.copy()
        if x.ndim == 1:
            x = np.array([x])

        if x.shape[1] != self.param_dim: raise AttributeError(
            "The input variable has dimension x.shape[1]==%d, while self.param_dim==%d" % (
            x.shape[1], self.param_dim))

        num_of_points = x.shape[0]

        output = np.empty( (num_of_points, ) + self.space_shape )

        mat_cache = [{} for i in range(self.param_dim)]
        interpolation_matices = [None] * self.param_dim

        # Linear Interpolation
        if self.interpolation_type == LINEAR_INTERPOLATION:
            for ((point, TTapp), (_, Xs)) in \
                    zip(np.ndenumerate(self.TTapprox), np.ndenumerate(self.Xs_params)):
                for i in range(self.param_dim):
                    try:
                        interpolation_matices[i] = mat_cache[i][tuple(Xs[i])]
                    except KeyError:
                        mat_cache[i][tuple(Xs[i])] = LinearInterpolationMatrix(Xs[i], x[:, i])
                        interpolation_matices[i] = mat_cache[i][tuple(Xs[i])]
                # TTval = TTapp.interpolate(interpolation_matices, is_sparse=is_sparse)
                TTval = interpolate(TTapp, interpolation_matices)
                output[(slice(None, None, None),) + point] = np.asarray(
                    [TTval[tuple([i] * self.param_dim)] for i in range(num_of_points)])
        elif self.interpolation_type == LAGRANGE_INTERPOLATION:  # Lagrange Interpolation
            for ((point, TTapp), (_, Xs), (_, bw)) in \
                    zip(np.ndenumerate(self.TTapprox), np.ndenumerate(self.Xs_params),
                        np.ndenumerate(self.barycentric_weights)):
                for i in range(self.param_dim):
                    try:
                        interpolation_matices[i] = mat_cache[i][tuple(Xs[i])]
                    except KeyError:
                        mat_cache[i][tuple(Xs[i])] = LagrangeInterpolationMatrix(Xs[i], bw[i], x[:, i])
                        interpolation_matices[i] = mat_cache[i][tuple(Xs[i])]
                TTval = interpolate(TTapp, interpolation_matices)
                output[(slice(None, None, None),) + point] = np.asarray(
                    [TTval[tuple([i] * self.param_dim)] for i in range(num_of_points)])
        else:
            raise AttributeError("Type of interpolation not defined")

        if orig_ndim == 1:
            if self.range_dim == 0:
                return output[(0,) + tuple([slice(None, None, None)] * self.range_dim)][0]
            else:
                return output[(0,) + tuple([slice(None, None, None)] * self.range_dim)]
        else:
            if self.range_dim == 0:
                return output[:, 0]
            else:
                return output


def LinearShapeFunction(x, xm, xp, xi):
    """ Hat function used for linear interpolation

    :param array x: 1d original points
    :param float xm,xp: bounding points of the support of the shape function
    :param array xi: 1d interpolation points

    :returns array N: evaluation of the shape function on xi
    """
    N = np.zeros(len(xi))
    if x != xm: N += (xi - xm) / (x - xm) * ((xi >= xm) * (xi <= x)).astype(float)
    if x != xp: N += ((x - xi) / (xp - x) + 1.) * ((xi >= x) * (xi <= xp)).astype(float)
    return N


def LinearInterpolationMatrix(x, xi):
    """
    LinearInterpolationMatrix(): constructs the Linear Interpolation Matrix from points ``x`` to points ``xi``

    Syntax:
        ``T = LagrangeInterpolationMatrix(x, xi)``

    Input:
        * ``x`` = (1d-array,float) set of ``N`` original points
        * ``xi`` = (1d-array,float) set of ``M`` interpolating points

    Output:
        * ``T`` = (2d-array(``MxN``),float) Linear Interpolation Matrix

    """

    M = np.zeros((len(xi), len(x)))

    M[:, 0] = LinearShapeFunction(x[0], x[0], x[1], xi)
    M[:, -1] = LinearShapeFunction(x[-1], x[-2], x[-1], xi)
    for i in range(1, len(x) - 1):
        M[:, i] = LinearShapeFunction(x[i], x[i - 1], x[i + 1], xi)

    return M


def BarycentricWeights(x):
    """
    BarycentricWeights(): Returns a 1-d array of weights for Lagrange Interpolation

    Syntax:
        ``w = BarycentricWeights(x)``

    Input:
        * ``x`` = (1d-array,float) set of points

    Output:
        * ``w`` = (1d-array,float) set of barycentric weights

    Notes:
        Algorithm (30) from :cite:`Kopriva2009`
    """
    N = x.shape[0]
    w = np.zeros((N))
    for j in range(0,N):
        w[j] = 1.
    for j in range(1,N):
        for k in range(0,j):
            w[k] = w[k] * (x[k] - x[j])
            w[j] = w[j] * (x[j] - x[k])
    for j in range(0,N):
        w[j] = 1. / w[j]
    return w

def LagrangeInterpolationMatrix(x, w, xi):
    """
    LagrangeInterpolationMatrix(): constructs the Lagrange Interpolation Matrix from points ``x`` to points ``xi``

    Syntax:
        ``T = LagrangeInterpolationMatrix(x, w, xi)``

    Input:
        * ``x`` = (1d-array,float) set of ``N`` original points
        * ``w`` = (1d-array,float) set of ``N`` barycentric weights
        * ``xi`` = (1d-array,float) set of ``M`` interpolating points

    Output:
        * ``T`` = (2d-array(``MxN``),float) Lagrange Interpolation Matrix

    Notes:
        Algorithm (32) from :cite:`Kopriva2009`
    """
    N = x.shape[0]
    M = xi.shape[0]
    T = np.zeros((M,N))
    for k in range(0,M):
        rowHasMatch = False
        for j in range(0,N):
            T[k,j] = 0.
            if np.isclose(xi[k],x[j]):
                rowHasMatch = True
                T[k,j] = 1.
        if (rowHasMatch == False):
            s = 0.
            for j in range(0,N):
                t = w[j] / (xi[k] - x[j])
                T[k,j] = t
                s = s + t
            for j in range(0,N):
                T[k,j] = T[k,j] / s
    return T


def interpolate(tensor, Ms=None, eps=1e-8, is_sparse=None):
    """ Interpolates the values of the TTvec at arbitrary points, using the interpolation matrices ``Ms``.

    :param list Ms: list of interpolation matrices for each dimension. Ms[i].shape[1] == tensor.shape()[i]
    :param float eps: tolerance with which to perform the rounding after interpolation
    :param list is_sparse: is_sparse[i] is a bool indicating whether Ms[i] is sparse or not. If 'None' all matrices are non sparse

    :returns: TTvec interpolation
    :rtype: TTvec


    """

    if len(Ms) != tensor.ndim():
        raise AttributeError("The length of Ms and the dimension of the TTvec must be the same!")

    d = len(Ms)
    for i in range(d):
        if Ms[i].shape[1] != tensor.shape()[i]:
            raise AttributeError("The condition  Ms[i].shape[1] == tensor.shape()[i] must hold.")


    if is_sparse == None: is_sparse = [False] * len(Ms)

    # Construct the interpolating TTmat
    TT_MND = TTmat(Ms[0].flatten(), nrows=Ms[0].shape[0], ncols=Ms[0].shape[1], is_sparse=[is_sparse[0]]).build()
    for M, s in zip(Ms[1:], is_sparse[1:]):
        TT_MND.kron(TTmat(M.flatten(), nrows=M.shape[0], ncols=M.shape[1], is_sparse=[s]).build())

    # Perform interpolation
    return TT_MND.dot(tensor).rounding(eps)
