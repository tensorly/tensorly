import numpy as np
from ..random import random_cp
from ._base_decomposition import DecompositionMixin
import tensorly as tl
from ..cp_tensor import CPTensor, validate_cp_rank, unfolding_dot_khatri_rao
from ..metrics import loss_operator, gradient_operator
from . import sample_khatri_rao


def initialize_generalized_parafac(tensor, rank, init='svd', svd='numpy_svd', loss=None, random_state=None):
    r"""Initialize factors used in `generalized parafac`.

    Parameters
    ----------
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices with uniform distribution using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor. If init is a previously initialized `cp tensor`, all
    the weights are pulled in the last factor and then the weights are set to "1" for the output tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    loss : {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}
    random_state : {None, int, np.random.RandomState}
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    rng = tl.check_random_state(random_state)
    if init == 'random':
        kt = random_cp(tl.shape(tensor), rank, random_state=rng, normalise_factors=False, **tl.context(tensor))

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                      svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, S, _ = svd_fun(tl.unfold(tensor, mode), n_eigenvecs=rank)

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, tl.shape(S)[0])
                U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

            if tensor.shape[mode] < rank:
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)

            factors.append(U[:, :rank])
        kt = CPTensor((None, factors))
    elif isinstance(init, (tuple, list, CPTensor)):
        try:
            weights, factors = CPTensor(init)

            if tl.all(weights == 1):
                weights, factors = CPTensor((None, factors))
            else:
                weights_avg = tl.prod(weights) ** (1.0 / tl.shape(weights)[0])
                for i in range(len(factors)):
                    factors[i] = factors[i] * weights_avg
            kt = CPTensor((None, factors))
            return kt
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a CPTensor instance'
            )
    else:
        raise ValueError('Initialization method "{}" not recognized'.format(init))
    if loss == 'gamma' or loss == 'rayleigh' or loss == 'poisson_count' or loss == 'bernoulli_odds':
        kt.factors = [tl.abs(f) for f in kt[1]]
    return kt


def stochastic_gradient(tensor, factors, batch_size, loss='gaussian', random_state=None, mask=None):
    '''
    Computes stochastic gradient between given tensor and estimated factors according to the given loss and batch size.

    Parameters
    ----------
    tensor : ndarray
    factors :  list of matrices
    batch_size : int
    loss : {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}
    random_state : {None, int, np.random.RandomState}
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.

    Returns
    -------
    ndarray
          Stochastic gradient between tensor and factors according to the given batch size and loss.
    '''
    if random_state is None or not isinstance(random_state, np.random.RandomState):
        rng = tl.check_random_state(random_state)
    else:
        rng = random_state
    if mask is None:
        indices_tuple = tuple([rng.randint(0, tl.shape(f)[0], size=batch_size, dtype=int) for f in factors])
    else:
        indices = list(tl.where(mask == 1))
        indices_to_select = tuple([rng.randint(0, tl.shape(indices[0]), size=batch_size, dtype=int)])
        indices_tuple = tuple([indices[i][indices_to_select] for i in range(len(indices))])
    modes = tl.ndim(tensor)
    gradient = [tl.zeros(tl.shape(factors[i])) for i in range(modes)]

    for mode in range(modes):
        indices_list = list(indices_tuple)
        indice_list_mode = indices_list.pop(mode)
        indices_list_mode_rest = indices_list.copy()
        indices_list.insert(mode, slice(None, None, None))
        indices_list = tuple(indices_list)
        sampled_kr, _ = sample_khatri_rao(factors, indices_list=indices_list_mode_rest, n_samples=batch_size,
                                          skip_matrix=mode)
        if mode:
            gradient_tensor = gradient_operator(tensor[indices_list], tl.dot(sampled_kr, tl.transpose(factors[mode])),
                                                loss=loss, mask=mask)
        else:
            gradient_tensor = gradient_operator(tl.transpose(tensor[indices_list]),
                                                tl.dot(sampled_kr, tl.transpose(factors[mode])), loss=loss, mask=mask)

        gradient[mode] = tl.index_update(gradient[mode], tl.index[indice_list_mode, :],
                                         tl.transpose(tl.dot(tl.transpose(sampled_kr), gradient_tensor))[indice_list_mode, :])
    return gradient


def generalized_parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd', tol=1e-8, verbose=0,
                        random_state=None, return_errors=False, loss='gaussian', lr=0.01, mask=None,
                        cvg_criterion='abs_rec_error'):
    """ Generalized PARAFAC decomposition by using FISTA optimization
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].
    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    lr : float
        Default : 0.01
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}
        Default : 'gaussian'
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None.
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)
    rng = tl.check_random_state(random_state)
    rec_errors = []
    modes = [mode for mode in range(tl.ndim(tensor))]
    # Initial tensor
    weights, factors = initialize_generalized_parafac(tensor, rank, init=init, svd=svd, loss=loss,
                                                      random_state=rng)
    gradient = []
    # Fista parameters
    momentum_old = tl.tensor(1.0)
    factors_update = []
    factors_new = []
    for i in modes:
        gradient.append(tl.zeros(tl.shape(factors[i])))
        factors_update.append((factors[i]))
        factors_new.append(tl.zeros(tl.shape(factors[i])))
    for iteration in range(n_iter_max):
        gradient_tensor = gradient_operator(tensor, tl.cp_to_tensor((None, factors)), loss=loss, mask=mask)
        for mode in modes:
            gradient[mode] = unfolding_dot_khatri_rao(gradient_tensor, (None, factors), mode)
            factors_new[mode] = factors_update[mode] - lr * gradient[mode]
            momentum = (1 + tl.sqrt(1 + 4 * momentum_old ** 2)) / 2
            factors_update[mode] = factors_new[mode] + ((momentum_old - 1) / momentum) * (factors_new[mode] - factors[mode])
            momentum_old = momentum
            factors[mode] = tl.copy(factors_new[mode])
            if loss == 'gamma' or loss == 'rayleigh' or loss == 'poisson_count' or loss == 'bernoulli_odds':
                factors[mode] = tl.clip(factors[mode], 0)
        # Calculate the current error
        current_loss = loss_operator(tensor, tl.cp_to_tensor((weights, factors)), loss=loss, mask=mask)
        rec_error = tl.sum(current_loss) / tl.norm(tensor)
        rec_errors.append(rec_error)

        if tol:
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]
                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}".format(iteration,
                                                                                         rec_error,
                                                                                         rec_error_decrease))
                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")
                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))
    cp_tensor = CPTensor((weights, factors))
    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor


def stochastic_generalized_parafac(tensor, rank, n_iter_max=1000, init='random', return_errors=False,
                                   loss='gaussian', epochs=20, batch_size=200, lr=0.01, beta_1=0.9, beta_2=0.999,
                                   mask=None, random_state=None):
    """ Generalized PARAFAC decomposition by using ADAM optimization
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].
    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}
        Default : 'gaussian'
    epochs : int
        Default : 20
    batch_size : int
        Default : 200
    lr : float
        Default : 0.01
    beta_1 : float
        ADAM optimization parameter.
        Default : 0.9
    beta_2 : float
        ADAM optimization parameter.
        Default : 0.999
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.
    random_state : {None, int, np.random.RandomState}

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)
    rng = tl.check_random_state(random_state)
    rec_errors = []
    modes = [mode for mode in range(tl.ndim(tensor))]
    # initial tensor
    _, factors = initialize_generalized_parafac(tensor, rank, init=init, loss=loss, random_state=rng)
    # parameters for ADAM optimization
    momentum_first = []
    momentum_second = []
    t_iter = 1
    indices_tuple = tuple([rng.randint(0, tl.shape(f)[0], size=batch_size, dtype=int) for f in factors])
    current_loss = loss_operator(tensor[indices_tuple],
                                 tl.sum(sample_khatri_rao(factors, indices_list=indices_tuple, n_samples=batch_size)[0],
                                 axis=1), loss=loss, mask=mask)
    # global loss
    current_loss = tl.sum(current_loss)
    for i in modes:
        momentum_first.append(tl.zeros(tl.shape(factors[i])))
        momentum_second.append(tl.zeros(tl.shape(factors[i])))
    epsilon = 1e-8
    bad_epochs = 0
    max_bad_epochs = 20
    for epoch in range(epochs):
        loss_old = tl.copy(current_loss)
        factors_old = [tl.copy(f) for f in factors]
        momentum_first_old = [tl.copy(f) for f in momentum_first]
        momentum_second_old = [tl.copy(f) for f in momentum_second]
        for iteration in range(n_iter_max):
            gradient = stochastic_gradient(tensor, factors, batch_size, random_state=rng, loss=loss)
            for mode in modes:
                # adam optimization
                momentum_first[mode] = (beta_1 * momentum_first[mode]) + (1 - beta_1) * gradient[mode]
                momentum_second[mode] = beta_2 * momentum_second[mode] + (1 - beta_2) * (gradient[mode] ** 2)
                momentum_first_hat = momentum_first[mode] / (1 - (beta_1 ** t_iter))
                momentum_second_hat = momentum_second[mode] / (1 - (beta_2 ** t_iter))
                factors[mode] = factors[mode] - lr * momentum_first_hat / (tl.sqrt(momentum_second_hat) + epsilon)

                if loss == 'gamma' or loss == 'rayleigh' or loss == 'poisson_count' or loss == 'bernoulli_odds':
                    factors[mode] = tl.clip(factors[mode], 0)

            t_iter += 1
        # Calculate the current error
        current_loss = loss_operator(tensor[indices_tuple],
                                     tl.sum(sample_khatri_rao(factors, indices_list=indices_tuple, n_samples=batch_size)[0],
                                     axis=1), loss=loss, mask=mask)
        # global loss
        current_loss = tl.sum(current_loss)
        if current_loss >= loss_old:
            lr = lr / 10
            factors = [tl.copy(f) for f in factors_old]
            current_loss = tl.copy(loss_old)
            t_iter -= iteration
            momentum_first = [tl.copy(f) for f in momentum_first_old]
            momentum_second = [tl.copy(f) for f in momentum_second_old]
            bad_epochs += 1
        else:
            bad_epochs = 0
        rec_error = current_loss / tl.norm(tensor)
        rec_errors.append(rec_error)
        if bad_epochs >= max_bad_epochs:
            print("Sufficient number of bad epochs")
            break
    cp_tensor = CPTensor((None, factors))
    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor


class GCP(DecompositionMixin):
    """ Generalized PARAFAC decomposition by using Fista optimization
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].
    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    lr : float
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None.
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """

    def __init__(self, rank, n_iter_max=100, init='svd', svd='numpy_svd', tol=1e-8, verbose=0, loss='gaussian', lr=0.01,
                 cvg_criterion='abs_rec_error', random_state=None, mask=None, return_errors=False):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.tol = tol
        self.verbose = verbose
        self.return_errors = return_errors
        self.loss = loss
        self.lr = lr
        self.random_state = random_state
        self.mask = mask
        self.cvg_criterion = cvg_criterion

    def fit_transform(self, tensor):
        """Decompose an input tensor
        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose
        Returns
        -------
        CPTensor
            decomposed tensor
        """
        cp_tensor, errors = generalized_parafac(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            tol=self.tol,
            verbose=self.verbose,
            loss=self.loss,
            lr=self.lr,
            random_state=self.random_state,
            mask=self.mask,
            return_errors=self.return_errors,
            cvg_criterion=self.cvg_criterion
        )
        self.decomposition_ = cp_tensor
        self.errors_ = errors
        return self.decomposition_


class Stochastic_GCP(DecompositionMixin):
    """ Stochastic Generalized PARAFAC decomposition by using ADAM optimization
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].
    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    lr : float
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """

    def __init__(self, rank, n_iter_max=100, init='random', loss='gaussian', epochs=100, batch_size=100, lr=0.01,
                 beta_1=0.9, beta_2=0.999, return_errors=False, random_state=None, mask=None):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.return_errors = return_errors
        self.loss = loss
        self.random_state = random_state
        self.lr = lr
        self.mask = mask

    def fit_transform(self, tensor):
        """Decompose an input tensor
        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose
        Returns
        -------
        CPTensor
            decomposed tensor
        """
        cp_tensor, errors = stochastic_generalized_parafac(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            epochs=self.epochs,
            batch_size=self.batch_size,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            loss=self.loss,
            lr=self.lr,
            random_state=self.random_state,
            mask=self.mask,
            return_errors=self.return_errors
        )
        self.decomposition_ = cp_tensor
        self.errors_ = errors
        return self.decomposition_
