from ..tenalg import khatri_rao, multi_mode_dot
from ..cp_tensor import CPTensor
from .. import backend as T
from .. import unfold, tensor_to_vec
from ..decomposition._cp import parafac

# Author: Cyrillus Tan, Jackson Chin, Aaron Meyer

# License: BSD 3 clause


class CP_PLSR:
    """CP tensor regression

        Learns a low rank CP tensor weight, This performs a partial least square regression to a tensor X (>= 2 modes)
        against a matrix Y. The first modes of X and Y will be considered coupled, and the decomposition will maximize
        the covariance between them.

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

    References
    ----------
    .. [1] Rasmus Bro, "Multiway calibration. Multilinear PLS", Chemometrics, 1996
    """

    def __init__(
        self, n_components, tol=1.0e-9, n_iter_max=100, random_state=None, verbose=False
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
            raise ValueError(
                "The first modes of X and Y must be coupled and have the same length."
            )
        X, Y = T.copy(X), T.copy(Y)

        # Check the shape of X and Y; convert vector Y to a matrix
        if T.ndim(X) < 2:
            raise ValueError("X must be at least a 2-mode tensor.")
        if (T.ndim(Y) != 1) and (T.ndim(Y) != 2):
            raise ValueError("Only a matrix (2-mode tensor) Y is allowed.")
        if T.ndim(Y) == 1:
            Y = T.reshape(Y, (-1, 1))

        # Mean center the data, record info the object
        self.X_shape_ = T.shape(X)
        self.Y_shape_ = T.shape(Y)
        self.X_mean_ = T.mean(X, axis=0)
        self.Y_mean_ = T.mean(Y, axis=0)
        X -= self.X_mean_
        Y -= self.Y_mean_

        self.X_factors = [
            T.zeros((l, self.n_components), **T.context(X)) for l in T.shape(X)
        ]
        self.Y_factors = [
            T.zeros((l, self.n_components), **T.context(X)) for l in T.shape(Y)
        ]

        ## FITTING EACH COMPONENT
        for component in range(self.n_components):
            comp_X_factors = [ff[:, component] for ff in self.X_factors]
            comp_Y_factors_0 = Y[:, 0]
            old_comp_Y_factors_0 = T.ones(T.shape(comp_Y_factors_0)) * T.inf

            for iter in range(self.n_iter_max):
                Z = T.tensordot(X, comp_Y_factors_0, axes=((0,), (0,)))

                if T.ndim(Z) >= 2:
                    Z_comp = parafac(
                        Z,
                        1,
                        tol=self.tol,
                        init="svd",
                        svd="randomized_svd",
                        normalize_factors=True,
                    )[1]
                else:
                    Z_comp = [Z / T.norm(Z)]

                for mode in range(
                    1, X.ndim
                ):  # Mode 0 of Z collapsed by above tensordot
                    comp_X_factors[mode] = tensor_to_vec(Z_comp[mode - 1])

                comp_X_factors[0] = multi_mode_dot(
                    X, comp_X_factors[1:], range(1, T.ndim(X))
                )
                comp_Y_factors_1 = T.dot(T.transpose(Y), comp_X_factors[0])
                comp_Y_factors_1 /= T.norm(comp_Y_factors_1)
                comp_Y_factors_0 = T.dot(Y, comp_Y_factors_1)

                if T.norm(old_comp_Y_factors_0 - comp_Y_factors_0) < self.tol:
                    if self.verbose:
                        print(
                            f"Component {component}: converged after {iter} iterations"
                        )
                    break
                old_comp_Y_factors_0 = T.copy(comp_Y_factors_0)

            # Put iteration results back to the parameter variables
            for ii in range(len(comp_X_factors)):
                self.X_factors[ii] = T.index_update(
                    self.X_factors[ii], T.index[:, component], comp_X_factors[ii]
                )
            self.Y_factors[0] = T.index_update(
                self.Y_factors[0], T.index[:, component], comp_Y_factors_0
            )
            self.Y_factors[1] = T.index_update(
                self.Y_factors[1], T.index[:, component], comp_Y_factors_1
            )

            # Deflation
            X -= CPTensor(
                (None, [T.reshape(ff, (-1, 1)) for ff in comp_X_factors])
            ).to_tensor()
            Y -= T.dot(
                T.dot(
                    self.X_factors[0],
                    T.lstsq(self.X_factors[0], T.reshape(comp_Y_factors_0, (-1, 1)))[0],
                ),
                T.reshape(comp_Y_factors_1, (1, -1)),
            )  # Y -= T pinv(T) u q'

        return self

    def predict(self, X):
        """Returns the predicted labels for a new data tensor

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, N1, ..., NS)
        """
        if self.X_shape_[1:] != T.shape(X)[1:]:
            raise ValueError(
                f"Training X has shape {self.X_shape_}, while the new X has shape {T.shape(X)}"
            )
        X -= self.X_mean_
        factors_kr = khatri_rao(self.X_factors, skip_matrix=0)
        unfolded = unfold(X, 0)
        scores = T.lstsq(factors_kr, T.transpose(unfolded))[0]  # = Tnew
        estimators = T.lstsq(self.X_factors[0], self.Y_factors[0])[0]
        return (
            T.dot(
                T.dot(T.transpose(scores), estimators), T.transpose(self.Y_factors[1])
            )
            + self.Y_mean_
        )

    def transform(self, X, Y=None):
        """Apply the dimension reduction from fitting to a new tensor.

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
        if self.X_shape_[1:] != T.shape(X)[1:]:
            raise ValueError(
                f"Training X has shape {self.X_shape_}, while the new X has shape {T.shape(X)}"
            )
        X = T.copy(X)
        X -= self.X_mean_
        X_scores = T.zeros((T.shape(X)[0], self.n_components), **T.context(X))

        for component in range(self.n_components):
            X_scores = T.index_update(
                X_scores,
                T.index[:, component],
                multi_mode_dot(
                    X,
                    [ff[:, component] for ff in self.X_factors[1:]],
                    range(1, T.ndim(X)),
                ),
            )
            X -= CPTensor(
                (
                    None,
                    [T.reshape(X_scores[:, component], (-1, 1))]
                    + [
                        T.reshape(ff[:, component], (-1, 1))
                        for ff in self.X_factors[1:]
                    ],
                )
            ).to_tensor()

        if Y is not None:
            Y = T.copy(Y)
            # Check on the shape of Y
            if (T.ndim(Y) != 1) and (T.ndim(Y) != 2):
                raise ValueError("Only a matrix (2-mode tensor) Y is allowed.")
            if T.ndim(Y) == 1:
                Y = T.reshape(Y, (-1, 1))
            if self.Y_shape_[1:] != T.shape(Y)[1:]:
                raise ValueError(
                    f"Training Y has shape {self.Y_shape_}, while the new Y has shape {T.shape(Y)}"
                )

            Y -= self.Y_mean_
            Y_scores = T.zeros((T.shape(Y)[0], self.n_components), **T.context(X))
            for component in range(self.n_components):
                Y_scores = T.index_update(
                    Y_scores,
                    T.index[:, component],
                    T.dot(Y, self.Y_factors[1][:, component]),
                )

                Y -= T.dot(
                    T.dot(
                        T.lstsq(T.transpose(X_scores), T.transpose(X_scores))[0],
                        Y_scores[:, [component]],
                    ),
                    T.transpose(self.Y_factors[1][:, [component]]),
                )  # Y -= T pinv(T) u q'
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
