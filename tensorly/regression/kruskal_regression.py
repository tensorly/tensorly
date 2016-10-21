from scipy.linalg import solve
import numpy as np
from ..base import partial_tensor_to_vec, partial_unfold
from ..tenalg import norm,  khatri_rao
from ..kruskal import kruskal_to_tensor, kruskal_to_vec
from ..utils import check_random_state

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


class KruskalRegressor():
    def __init__(self, weight_rank, tol=10e-7, reg_W=1, n_iter_max=100, random_state=None, verbose=1):
        """Kruskal tensor regression

            Learns a low rank CP tensor weight

        Parameters
        ----------
        weight_rank : int
            rank of the CP decomposition of the regression weights
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
        self.weight_rank = weight_rank
        self.tol = tol
        self.reg_W = reg_W
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters
        """
        params = ['weight_rank', 'tol', 'reg_W', 'n_iter_max', 'random_state', 'verbose']
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
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        y : 1D-array of shape (n_samples, )
            labels associated with each sample

        Returns
        -------
        self
        """
        rng = check_random_state(self.random_state)

        # Initialise randomly the weights
        W = []
        for i in range(1, X.ndim):  # The first dimension of X is the number of samples
            W.append(rng.randn(X.shape[i], self.weight_rank))

        # Norm of the weight tensor at each iteration
        norm_W = []

        for iteration in range(self.n_iter_max):

            # Optimise each factor of W
            for i in range(len(W)):
                phi = partial_unfold(X, i, skip_begin=1).dot(khatri_rao(W, skip_matrix=i)).reshape((X.shape[0], -1))
                inv_term = np.dot(phi.T, phi) + self.reg_W*np.eye(phi.shape[1])
                W[i] = np.reshape(solve(inv_term, phi.T.dot(y)), (X.shape[i + 1], self.weight_rank))

            weight_tensor_ = kruskal_to_tensor(W)
            norm_W.append(norm(weight_tensor_, 2))

            # Convergence check
            if iteration > 1:
                weight_evolution = abs(norm_W[-1] - norm_W[-2]) / norm_W[-1]

                if (weight_evolution <= self.tol):
                    if self.verbose:
                        print('\nConverged in {} iterations'.format(iteration))
                    break

        self.weight_tensor_ = weight_tensor_
        self.kruskal_weight_ = W
        self.vec_W_ = kruskal_to_vec(W)
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
        return np.dot(partial_tensor_to_vec(X), self.vec_W_)
