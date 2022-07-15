import numpy as np
from ..base import partial_unfold
from ..tenalg import khatri_rao
from ..cp_tensor import cp_to_tensor, cp_to_vec
from .. import backend as T

# Author: Cyrillus Tan, Jackson Chin, Aaron Meyer

# License: BSD 3 clause


class CP_PLSR():
    """CP tensor regression

        Learns a low rank CP tensor weight

    Parameters
    ----------
    n_components : int
        rank of the CP decomposition of the regression weights
    tol : float
        convergence value
    n_iter_max : int, optional, default is 100
        maximum number of iteration
    random_state : None, int or RandomState, optional, default is None
    verbose : bool, default is False
        whether to be verbose during fitting
    """

    def __init__(self, n_components, tol=1.0e-7, n_iter_max=100, random_state=None, verbose=False):
        self.n_components = n_components
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters
        """
        params = ['n_components', 'tol', 'n_iter_max', 'random_state', 'verbose']
        return {param_name: getattr(self, param_name) for param_name in params}

    def set_params(self, **parameters):
        """Sets the value of the provided parameters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, Y):
        """Fits the model to the data (X, Y)

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        Y : 2D-array of shape (n_samples, n_predictions)
            labels associated with each sample

        Returns
        -------
        self
        """
        # Check that both tensors are coupled along the first mode
        assert T.shape(X)[0] == T.shape(Y)[0]

        # Make Y 2D if it is a vector
        if T.ndim(Y) == 1:
            Y = T.reshape(Y, (-1, 1))

        raise NotImplementedError

    def predict(self, X):
        """Returns the predicted labels for a new data tensor

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        """
        raise NotImplementedError

    def transform(self, X, Y=None):
        """Apply the dimension reduction.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.
        Y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.
        Returns
        -------
        x_scores, y_scores : array-like or tuple of array-like
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        raise NotImplementedError

    def fit_transform(self, X, Y=None):
        """Learn and apply the dimension reduction on the train data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.
        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.
        Returns
        -------
        self : ndarray of shape (n_samples, n_components)
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        return self.fit(X, Y).transform(X, Y)
