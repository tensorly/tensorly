import tensorly as tl
import math
from ..base import partial_tensor_to_vec, partial_unfold
from ..tenalg import khatri_rao
from ..cp_tensor import cp_to_tensor, cp_to_vec
from .. import backend as T

# Author: Jean Kossaifi

# License: BSD 3 clause


class CPRegressor:
    r"""CP tensor regression

        Learns a low rank CP tensor weight

    Parameters
    ----------
    weight_rank : int
        rank of the CP decomposition of the regression weights
    tol : float
        convergence value
    reg_W : float, optional, default is 1
        l2 regularisation constant for the regression weights (:math:`reg_W * \sum_i ||factors[i]||_F^2`)
    n_iter_max : int, optional, default is 100
        maximum number of iteration
    random_state : None, int or RandomState, optional, default is None
    verbose : int, default is 1
        level of verbosity
    """

    def __init__(
        self,
        weight_rank,
        tol=10e-7,
        reg_W=1,
        n_iter_max=100,
        random_state=None,
        verbose=1,
    ):
        self.weight_rank = weight_rank
        self.tol = tol
        self.reg_W = reg_W
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, **kwargs):
        """Returns a dictionary of parameters"""
        params = [
            "weight_rank",
            "tol",
            "reg_W",
            "n_iter_max",
            "random_state",
            "verbose",
        ]
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
        X : tensor data of shape (n_samples, I_1, ..., I_p)
        y : tensor of shape (n_samples, O_1, ..., O_q)
            labels associated with each sample

        Returns
        -------
        self
        """
        rng = T.check_random_state(self.random_state)

        # Initialise the weights randomly
        W = []
        for i in range(1, T.ndim(X)):  # The first dimension is the number of samples
            W.append(T.tensor(rng.randn(X.shape[i], self.weight_rank), **T.context(X)))
        for i in range(1, T.ndim(y)):
            W.append(T.tensor(rng.randn(y.shape[i], self.weight_rank), **T.context(X)))

        # Norm of the weight tensor at each iteration
        norm_W = []
        weights = T.ones(self.weight_rank, **T.context(X))

        for iteration in range(self.n_iter_max):
            # Optimise each factor of W
            for i in range(len(W)):
                if i < T.ndim(X) - 1:
                    X_unfolded = partial_unfold(X, i, skip_begin=1)
                    phi = T.dot(
                        X_unfolded,
                        T.reshape(
                            khatri_rao(W, skip_matrix=i), (X_unfolded.shape[-1], -1)
                        ),
                    )
                    phi = T.transpose(
                        T.reshape(
                            phi, (X.shape[0], X.shape[i + 1], -1, self.weight_rank)
                        ),
                        (0, 2, 1, 3),
                    )
                    phi = T.reshape(phi, (-1, X.shape[i + 1] * self.weight_rank))
                    y_reshaped = T.reshape(y, (-1,))
                    inv_term = T.dot(T.transpose(phi), phi) + self.reg_W * T.eye(
                        phi.shape[1], **T.context(X)
                    )
                    W[i] = T.reshape(
                        T.solve(inv_term, T.dot(T.transpose(phi), y_reshaped)),
                        (-1, self.weight_rank),
                    )
                else:
                    X_unfolded = partial_tensor_to_vec(X, skip_begin=1)
                    phi = T.dot(
                        X_unfolded,
                        T.reshape(
                            khatri_rao(W, skip_matrix=i), (X_unfolded.shape[-1], -1)
                        ),
                    )
                    phi = T.reshape(phi, (-1, self.weight_rank))
                    y_reshaped = T.reshape(
                        T.moveaxis(y, i - T.ndim(X) + 2, -1),
                        (-1, y.shape[i - T.ndim(X) + 2]),
                    )
                    inv_term = T.dot(T.transpose(phi), phi) + self.reg_W * T.eye(
                        phi.shape[1], **T.context(X)
                    )
                    W[i] = T.transpose(
                        T.solve(inv_term, T.dot(T.transpose(phi), y_reshaped))
                    )

            weight_tensor_ = cp_to_tensor((weights, W))
            norm_W.append(T.norm(weight_tensor_, 2))

            # Convergence check
            if iteration > 1:
                weight_evolution = tl.abs(norm_W[-1] - norm_W[-2]) / norm_W[-1]

                if weight_evolution <= self.tol:
                    if self.verbose:
                        print(f"\nConverged in {iteration} iterations")
                    break

        self.weight_tensor_ = weight_tensor_
        self.cp_weight_ = (weights, W)

        self.vec_W_ = cp_to_vec((weights, W))
        self.n_iterations_ = iteration + 1
        self.norm_W_ = norm_W

        return self

    def predict(self, X):
        """Returns the predicted labels for a new data tensor

        Parameters
        ----------
        X : ndarray
            tensor data of shape (n_samples, I_1, ..., I_p)
        """
        out_shape = (-1, *self.weight_tensor_.shape[T.ndim(X) - 1 :])
        if T.ndim(self.weight_tensor_) > T.ndim(X) - 1:
            weight_shape = (
                -1,
                int(math.prod(self.weight_tensor_.shape[T.ndim(X) - 1 :])),
            )
        else:
            weight_shape = (-1,)
        return T.reshape(
            T.dot(
                partial_tensor_to_vec(X), T.reshape(self.weight_tensor_, weight_shape)
            ),
            out_shape,
        )
