import numpy as np
from ..base import unfold, vec_to_tensor
from ..base import partial_tensor_to_vec, partial_unfold
from ..tenalg import kronecker
from ..tucker_tensor import tucker_to_tensor, tucker_to_vec
from ..random import check_random_state
from .. import backend as T

# Author: Jean Kossaifi

# License: BSD 3 clause



class TuckerRegressor():
    """Tucker tensor regression

        Learns a low rank Tucker weight for the regression

    Parameters
    ----------
    weight_ranks : int list
        dimension of each mode of the core Tucker weight
    tol : float
        convergence value
    reg_W : int, optional, default is 1
        regularisation on the weights
    n_iter_max : int, optional, default is 100
        maximum number of iteration
    random_state : None, int or RandomState, optional, default is None
    verbose : int, default is 1
        level of verbosity
    """

    def __init__(self, weight_ranks, tol=10e-7, reg_W=1, n_iter_max=100, random_state=None, verbose=1):
        self.weight_ranks = weight_ranks
        self.tol = tol
        self.reg_W = reg_W
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters
        """
        params = ['weight_ranks', 'tol', 'reg_W', 'n_iter_max', 'random_state', 'verbose']
        return {param_name: getattr(self, param_name) for param_name in params}

    def set_params(self, **parameters):
        """Sets the value of the provided parameters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """Fits the model to the data (X, y)

        Parameters
        ----------
        X : ndarray of shape (n_samples, N1, ..., NS)
            tensor data
        y : array of shape (n_samples)
            labels associated with each sample

        Returns
        -------
        self
        """
        rng = check_random_state(self.random_state)

        # Initialise randomly the weights
        G = T.tensor(rng.randn(*self.weight_ranks), **T.context(X))
        W = []
        for i in range(1, T.ndim(X)):  # First dimension of X = number of samples
            W.append(T.tensor(rng.randn(X.shape[i], G.shape[i - 1]), **T.context(X)))

        # Norm of the weight tensor at each iteration
        norm_W = []

        for iteration in range(self.n_iter_max):

            # Optimise modes of W
            for i in range(len(W)):
                phi = partial_tensor_to_vec(
                            T.dot(partial_unfold(X, i),
                                  T.dot(kronecker(W, skip_matrix=i),
                                          T.transpose(unfold(G, i)))))
                # Regress phi on y: we could call a package here, e.g. scikit-learn
                inv_term = T.dot(T.transpose(phi), phi) +\
                     self.reg_W * T.tensor(np.eye(phi.shape[1]), **T.context(X))
                W_i = vec_to_tensor(T.solve(inv_term, T.dot(T.transpose(phi), y)),
                                    (X.shape[i + 1], G.shape[i]))
                W[i] = W_i

            phi = T.dot(partial_tensor_to_vec(X), kronecker(W))
            G = vec_to_tensor(T.solve(T.dot(T.transpose(phi), phi) +\
                                        self.reg_W * T.tensor(np.eye(phi.shape[1]), **T.context(X)),
                                      T.dot(T.transpose(phi), y)), G.shape)

            weight_tensor_ = tucker_to_tensor((G, W))
            norm_W.append(T.norm(weight_tensor_, 2))

            # Convergence check
            if iteration > 1:
                weight_evolution = abs(norm_W[-1] - norm_W[-2]) / norm_W[-1]

                if (weight_evolution <= self.tol):
                    if self.verbose:
                        print('\nConverged in {} iterations'.format(iteration))
                    break

        self.weight_tensor_ = weight_tensor_
        self.tucker_weight_ = (G, W)
        self.vec_W_ = tucker_to_vec((G, W))
        self.n_iterations_ = iteration + 1
        self.norm_W_ = norm_W

        return self

    def predict(self, X):
        """Returns the predicted labels for a new data tensor

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        """
        return T.dot(partial_tensor_to_vec(X), self.vec_W_)
