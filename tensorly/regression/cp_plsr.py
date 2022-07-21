from ..tenalg import khatri_rao, mode_dot, multi_mode_dot
from ..cp_tensor import CPTensor, cp_to_tensor, cp_to_vec
from ..decomposition import tucker
from .. import backend as T
from .. import unfold

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

    def __init__(
        self, n_components, tol=1.0e-7, n_iter_max=100, random_state=None, verbose=False
    ):
        self.n_components = n_components
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters"""
        params = ["n_components", "tol", "n_iter_max", "random_state", "verbose"]
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
        ## PREPROCESSING
        # Check that both tensors are coupled along the first mode
        if T.shape(X)[0] != T.shape(Y)[0]:
            raise ValueError("The first modes of X and Y must be coupled and have the same length.")
        X, Y = T.copy(X), T.copy(Y)

        # Check the shape of X and Y; convert vector Y to a matrix
        if T.ndim(X) < 2:
            raise ValueError("X must be at least a 2-mode tensor.")
        if (T.ndim(Y) != 1) and (T.ndim(Y) != 2):
            raise ValueError("Only a matrix (2-mode tensor) Y is allowed.")
        if T.ndim(Y) == 1:
            Y = T.reshape(Y, (-1, 1))

        # Mean center the data, record info the object
        self.X_shape = T.shape(X)
        self.Y_shape = T.shape(Y)
        self.X_mean = T.mean(X, axis=0)
        self.Y_mean = T.mean(Y, axis=0)
        X -= self.X_mean
        Y -= self.Y_mean

        self.X_factors = [T.zeros((l, self.n_components)) for l in T.shape(X)]
        self.Y_factors = [T.tile(Y[:, [0]], self.n_components), T.zeros((T.shape(Y)[1], self.n_components))]

        ## FITTING EACH COMPONENT
        for a in range(self.n_components):
            oldU = T.ones(T.shape(self.Y_factors[0][:, a])) * T.inf
            for iter in range(self.n_iter_max):
                Z = T.einsum("i...,i...->...", X, self.Y_factors[0][:, a])
                Z_comp = tucker(Z, [1] * T.ndim(Z))[1] if Z.ndim >= 2 else [Z / T.norm(Z)]
                for ii in range(Z.ndim):
                    self.X_factors[ii + 1][:, a] = Z_comp[ii].flatten()

                self.X_factors[0][:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, T.ndim(X)))
                self.Y_factors[1][:, a] = T.dot(T.transpose(Y), self.X_factors[0][:, a])
                self.Y_factors[1][:, a] /= T.norm(self.Y_factors[1][:, a])
                self.Y_factors[0][:, a] = T.dot(Y, self.Y_factors[1][:, a])
                if T.norm(oldU - self.Y_factors[0][:, a]) < self.tol:
                    if self.verbose:
                        print("Component {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Y_factors[0][:, a].copy()

            X -= CPTensor((None, [ff[:, a].reshape(-1, 1) for ff in self.X_factors])).to_tensor()
            Y -= T.dot(T.dot(T.dot(self.X_factors[0], T.pinv(self.X_factors[0])),
                             self.Y_factors[0][:, [a]]), T.transpose(self.Y_factors[1][:, [a]]))   # Y -= T pinv(T) u q'

        return self


    def predict(self, X):
        """Returns the predicted labels for a new data tensor

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        """
        if self.X_shape[1:] != T.shape(X)[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {T.shape(X)}")
        X -= self.X_mean
        factors_kr = khatri_rao(self.X_factors, skip_matrix=0)
        unfolded = unfold(X, 0)
        scores = T.lstsq(factors_kr, T.transpose(unfolded), rcond=-1)[0]  # = Tnew
        estimators = T.lstsq(self.X_factors[0], self.Y_factors[0], rcond=-1)[0]
        return T.dot(T.dot(T.transpose(scores), estimators), T.transpose(self.Y_factors[1])) + self.Y_mean


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
        X_scores, Y_scores : array-like or tuple of array-like
            Return `X_scores` if `Y` is not given, `(X_scores, Y_scores)` otherwise.
        """
        if self.X_shape[1:] != T.shape(X)[1:]:
            raise ValueError(f"Training X has shape {self.X_shape}, while the new X has shape {T.shape(X)}")
        X = T.copy(X)
        X -= self.X_mean
        X_scores = T.zeros((T.shape(X)[0], self.n_components))

        for a in range(self.n_components):
            X_scores[:, a] = multi_mode_dot(X, [ff[:, a] for ff in self.X_factors[1:]], range(1, T.ndim(X)))
            X -= CPTensor((None, [X_scores[:, a].reshape(-1, 1)] + [ff[:, a].reshape(-1, 1) for ff in self.X_factors[1:]])).to_tensor()

        if Y is not None:
            Y = T.copy(Y)
            # Check on the shape of Y
            if (T.ndim(Y) != 1) and (T.ndim(Y) != 2):
                raise ValueError("Only a matrix (2-mode tensor) Y is allowed.")
            if T.ndim(Y) == 1:
                Y = T.reshape(Y, (-1, 1))
            if self.Y_shape[1:] != T.shape(Y)[1:]:
                raise ValueError(f"Training Y has shape {self.Y_shape}, while the new Y has shape {T.shape(Y)}")

            Y -= self.Y_mean
            Y_scores = T.zeros((T.shape(Y)[0], self.n_components))
            for a in range(self.n_components):
                Y_scores[:, a] = T.dot(Y, self.Y_factors[1][:, a])
                Y -= T.dot(T.dot(T.dot(X_scores, T.pinv(X_scores)), Y_scores[:, [a]]),
                           T.transpose(self.Y_factors[1][:, [a]]))  # Y -= T pinv(T) u q'
            return X_scores, Y_scores

        return X_scores


    def fit_transform(self, X, Y):
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
