import tensorly as tl
# from ..random import check_random_state
# rng = check_random_state(1)

import time
import random
import numpy as np


from TensorToolbox.core import TensorWrapper, TTvec
from SpectralToolbox import Spectral1D as S1D

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
                 eps=1e-4, method="ttcross",
                 rs=None, Jinit=None, delta=1e-4, maxit=100,
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

        self.surr_type = None  # Can be Projection or Interpolation
        self.barycentric_weights = None  # np.array of List of barycentric weights for Lagrange Interp

        self.range_dim = None  # Number of dimensions of the spatial space
        self.param_dim = None  # Number of dimensions of the parameter space
        self.TW = None  # Tensor wrapper containing caching the function evaluations
        self.params = None  # Parameters to be passed to f

        self.init = False  # Flags whether the construction of the approximation/surrogate is done.

        # Parameters to be reset on restarting
        self.f = None  # Function to be approximated


        # End list of attributes
        #########################################################

        self.f = f
        self.params = params
        self.surr_type = surrogate_type

        # Store all the tt approximation parameters
        self.method = method
        self.eps = eps
        self.rs = rs
        self.Jinit = Jinit
        self.delta = delta
        self.maxit = maxit
        self.mv_eps = mv_eps
        self.mv_maxit = mv_maxit

        self.range_dim = range_dim
        self.param_dim = len(grids) - self.range_dim
        if self.param_dim < 0: raise AttributeError("The grids argument must respect len(grids) >= range_dim")

        self.set_grids(grids)

        # Definition of the Tensor Wrapper (works for homogeneous grid on param space)
        self.TW = TensorWrapper(self.f,
                                self.Xs_params[tuple([0] * max(self.range_dim, 1))],
                                params=self.params
                                )

    def __call__(self, x_in, verbose=False):
        """ Evaluate the surrogate on points ``x_in``

        :param np.ndarray x_in: 1 or 2 dimensional array of points in the parameter space where to evaluate the function. In 2 dimensions, each row is an entry, i.e. ``x_in.shape[1] == self.param_dim``

        :return: an array with dimension equal to the space dimension (``range_dim``) plus one. If ``A`` is the returned vector and ``range_dim=2``, then ``A[i,:,:]`` is the value of the surrogate for ``x_in[i,:]``

        """
        if not self.init:
            raise RuntimeError(
                "The SpectralTensorTrain approximation is not initialized or is not set to construct a surrogate")
        else:
            if not isinstance(x_in, np.ndarray) or x_in.ndim not in [1, 2]: raise AttributeError(
                "The input variable must be a 1 or 2 dimensional numpy.ndarray")
            orig_ndim = x_in.ndim
            x = x_in.copy()
            if x.ndim == 1:
                x = np.array([x])

            if x.shape[1] != self.param_dim: raise AttributeError(
                "The input variable has dimension x.shape[1]==%d, while self.param_dim==%d" % (
                x.shape[1], self.param_dim))

            Np = x.shape[0]

            output = np.empty((Np,) + self.space_shape)

            mat_cache = [{} for i in range(self.param_dim)]
            MsI = [None] * self.param_dim

            if self.surr_type == LINEAR_INTERPOLATION:  # Linear Interpolation
                for ((point, TTapp), (_, Xs)) in \
                        zip(np.ndenumerate(self.TTapprox), np.ndenumerate(self.Xs_params)):
                    for i in range(self.param_dim):
                        try:
                            MsI[i] = mat_cache[i][tuple(Xs[i])]
                        except KeyError:
                            mat_cache[i][tuple(Xs[i])] = S1D.SparseLinearInterpolationMatrix(Xs[i], x[:, i]).tocsr()
                            MsI[i] = mat_cache[i][tuple(Xs[i])]
                    is_sparse = [True] * self.param_dim
                    TTval = TTapp.interpolate(MsI, is_sparse=is_sparse)
                    output[(slice(None, None, None),) + point] = np.asarray(
                        [TTval[tuple([i] * self.param_dim)] for i in range(Np)])
            elif self.surr_type == LAGRANGE_INTERPOLATION:  # Lagrange Interpolation
                for ((point, TTapp), (_, Xs), (_, bw)) in \
                        zip(np.ndenumerate(self.TTapprox), np.ndenumerate(self.Xs_params),
                            np.ndenumerate(self.barycentric_weights)):
                    for i in range(self.param_dim):
                        try:
                            MsI[i] = mat_cache[i][tuple(Xs[i])]
                        except KeyError:
                            mat_cache[i][tuple(Xs[i])] = S1D.LagrangeInterpolationMatrix(Xs[i], bw[i], x[:, i])
                            MsI[i] = mat_cache[i][tuple(Xs[i])]
                    is_sparse = [False] * self.param_dim
                    TTval = TTapp.interpolate(MsI, is_sparse=is_sparse)
                    output[(slice(None, None, None),) + point] = np.asarray(
                        [TTval[tuple([i] * self.param_dim)] for i in range(Np)])
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

    def set_grids(self, grids):
        # Store grid for spatial space
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

    def build(self):
        self.start_build_time = time.clock()

        if self.generic_approx is None:
            self.generic_approx = np.empty(self.space_shape, dtype=TTvec)

        for point, val in np.ndenumerate(self.generic_approx):
            if val is None or not val.init:
                # try:
                multidim_point = point if self.range_dim > 0 else None
                # Build generic_approx for the selected point
                if val is None or self.generic_approx[point].Jinit is None:
                    # Find all the back neighbors and select the first found to start from
                    neigh = None
                    for i in range(self.range_dim - 1, -1, -1):
                        if point[i] - 1 >= 0:
                            pp = list(point)
                            pp[i] -= 1
                            neigh = self.generic_approx[tuple(pp)]
                            break
                    # If a neighbor is found, select the rank and the fibers to be used
                    if neigh != None:
                        if self.method == 'ttcross':
                            rs = neigh.ranks()
                            for i in range(1, len(rs) - 1):
                                rs[i] += 1
                            Js = neigh.Js_last[-2]
                            # Trim the fibers according to rs (This allow the rank to decrease as well)
                            for r, (j, J) in zip(rs[1:-1], enumerate(Js)):
                                Js[j] = random.sample(J, r)
                            self.rs = rs
                            self.Jinit = Js

                    self.generic_approx[point] = TTvec(self.TW, multidim_point=multidim_point)
                    self.generic_approx[point].build(eps=self.eps, method=self.method,
                                                     rs=self.rs,
                                                     Jinit=self.Jinit,
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
        if self.surr_type == LAGRANGE_INTERPOLATION:
            if self.barycentric_weights is None:
                self.barycentric_weights = np.empty(self.space_shape, dtype=list)

            for (point, _), (_, X) in \
                    zip(np.ndenumerate(self.barycentric_weights), np.ndenumerate(self.Xs_params)):
                self.barycentric_weights[point] = [S1D.BarycentricWeights(X[i]) for i in range(self.param_dim)]

        self.init = True




