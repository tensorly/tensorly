import warnings
import tensorly as tl
from tensorly.random import random_parafac2
from tensorly import backend as T
from . import parafac
from ..parafac2_tensor import parafac2_to_slice, Parafac2Tensor, _validate_parafac2_tensor
from ..kruskal_tensor import kruskal_normalise, KruskalTensor
from ..base import unfold

# Authors: Marie Roald
#          Yngve Mardal Moe


def initialize_decomposition(tensor_slices, rank, init='random', svd='numpy_svd', random_state=None):
    r"""Initiate a random PARAFAC2 decomposition given rank and tensor slices

    Parameters
    ----------
    tensor_slices : Iterable of ndarray
    rank : int
    init : {'random', 'svd', KruskalTensor, Parafac2Tensor}, optional
    random_state : `np.random.RandomState`

    Returns
    -------
    parafac2_tensor : Parafac2Tensor
        List of initialized factors of the CP decomposition where element `i`
        is of shape (tensor.shape[i], rank)

    """
    context = tl.context(tensor_slices[0])
    shapes = [m.shape for m in tensor_slices]
    
    if init == 'random':
        return random_parafac2(shapes, rank, full=False, random_state=random_state,
                               **context)
    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                    svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)
        
        padded_tensor = _pad_by_zeros(tensor_slices)
        A = svd_fun(unfold(padded_tensor, 0), n_eigenvecs=rank)[0]
        C = svd_fun(unfold(padded_tensor, 2), n_eigenvecs=rank)[0]
        B = T.eye(rank, **context)
        projections = _compute_projections(tensor_slices, (A, B, C), svd_fun)
        return Parafac2Tensor((None, (A, B, C), projections))

    elif isinstance(init, (tuple, list, Parafac2Tensor, KruskalTensor)):
        try:
            decomposition = Parafac2Tensor.from_kruskaltensor(init, parafac2_tensor_ok=True)
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a Parafac2Tensor instance'
            )
        if decomposition.rank != rank:
            raise ValueError('Cannot init with a decomposition of different rank')
        return decomposition
    raise ValueError('Initialization method "{}" not recognized'.format(init))


def _pad_by_zeros(tensor_slices):
    """Return zero-padded full tensor.
    """
    I = len(tensor_slices)
    J = max(tensor_slice.shape[0] for tensor_slice in tensor_slices)
    K = tensor_slices[0].shape[1]
    padded = T.zeros((I, J, K), **T.context(tensor_slices[0])) 
    for i, tensor_slice in enumerate(tensor_slices):
        J_i = len(tensor_slice)
        
        tl.index_update(padded, tl.index[i, :J_i], tensor_slice)
    
    return padded


def _compute_projections(tensor_slices, factors, svd_fun, out=None):
    A, B, C = factors

    if out is None:
        out = [T.zeros((tensor_slice.shape[0], C.shape[1]), **T.context(tensor_slice)) for tensor_slice in tensor_slices]

    slice_idxes = range(T.shape(A)[0])
    for projection, i, tensor_slice in zip(out, slice_idxes, tensor_slices):
        a_i = A[i]
        lhs = T.dot(B, T.transpose(a_i*C))
        rhs = T.transpose(tensor_slice)
        U, S, Vh = svd_fun(T.dot(lhs, rhs), n_eigenvecs=A.shape[1])

        out[i] = tl.index_update(projection, tl.index[:], T.transpose(T.dot(U, Vh)))

    return out


def _project_tensor_slices(tensor_slices, projections, out=None):
    if out is None:
        rank = projections[0].shape[1]
        num_slices = len(tensor_slices)
        num_cols = tensor_slices[0].shape[1]
        out = T.zeros((num_slices, rank, num_cols), **T.context(tensor_slices[0]))

    for i, (tensor_slice, projection) in enumerate(zip(tensor_slices, projections)):
        slice_ = T.dot(T.transpose(projection), tensor_slice)
        out = tl.index_update(out, tl.index[i, :], slice_)
    return out


def _get_svd(svd):
    if svd in tl.SVD_FUNS:
        return tl.SVD_FUNS[svd]
    else:
        message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                svd, tl.get_backend(), tl.SVD_FUNS)
        raise ValueError(message)


def _parafac2_reconstruction_error(tensor_slices, decomposition):
    _validate_parafac2_tensor(decomposition)
    squared_error = 0
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx, validate=False)
        squared_error += tl.sum((tensor_slice - reconstruction)**2)
    return tl.sqrt(squared_error)


def parafac2(tensor_slices, rank, n_iter_max=100, init='random', svd='numpy_svd', normalize_factors=False,
             tol=1e-8, random_state=None, verbose=False, return_errors=False, n_iter_parafac=5):
    r"""PARAFAC2 decomposition [1]_ of a third order tensor via alternating least squares (ALS)

    Computes a rank-`rank` PARAFAC2 decomposition of the third-order tensor defined by 
    `tensor_slices`. The decomposition is on the form :math:`(A [B_i] C)` such that the
     i-th frontal slice, :math:`X_i`, of :math:`X` is given by

    .. math::
    
        X_i = B_i diag(a_i) C^T,
    
    where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`A`, :math:`B_i` 
    is a :math:`J_i \times R` factor matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
    is constant for all :math:`i`, and :math:`C` is a :math:`K \times R` factor matrix. 
    To compute this decomposition, we reformulate the expression for :math:`B_i` such that

    .. math::

        B_i = P_i B,

    where :math:`P_i` is a :math:`J_i \times R` orthogonal matrix and :math:`B` is a
    :math:`R \times R` matrix.

    An alternative formulation of the PARAFAC2 decomposition is that the tensor element
    :math:`X_{ijk}` is given by

    .. math::

        X_{ijk} = \sum_{r=1}^R A_{ir} B_{ijr} C_{kr},
    
    with the same constraints hold for :math:`B_i` as above.
     

    Parameters
    ----------
    tensor_slices : ndarray or list of ndarrays
        Either a third order tensor or a list of second order tensors that may have different number of rows.
        Note that the second mode factor matrices are allowed to change over the first mode, not the
        third mode as some other implementations use (see note below).
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', KruskalTensor, Parafac2Tensor}
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : bool (optional)
        If True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors. Note that
        there may be some inaccuracies in the component weights.
    tol : float, optional
        (Default: 1e-8) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    n_iter_parafac: int, optional
        Number of PARAFAC iterations to perform for each PARAFAC2 iteration

    Returns
    -------
    Parafac2Tensor : (weight, factors, projection_matrices)
        * weights : 1D array of shape (rank, )
            all ones if normalize_factors is False (default), 
            weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape
            (tensor.shape[i], rank)
        * projection_matrices : List of projection matrices used to create evolving
            factors.
         
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] Kiers, H.A.L., ten Berge, J.M.F. and Bro, R. (1999), 
            PARAFAC2—Part I. A direct fitting algorithm for the PARAFAC2 model. 
            J. Chemometrics, 13: 275-294.

    Notes
    -----
    This formulation of the PARAFAC2 decomposition is slightly different from the one in [1]_.
    The difference lies in that here, the second mode changes over the first mode, whereas in
    [1]_, the second mode changes over the third mode. We made this change since that means
    that the function accept both lists of matrices and a single nd-array as input without
    any reordering of the modes.
    """
    weights, factors, projections = initialize_decomposition(tensor_slices, rank, random_state=random_state)

    rec_errors = []
    norm_tensor = tl.sqrt(sum(tl.norm(tensor_slice, 2) for tensor_slice in tensor_slices))
    svd_fun = _get_svd(svd)

    projected_tensor = tl.zeros([factor.shape[0] for factor in factors], **T.context(factors[0]))

    for iteration in range(n_iter_max):
        if verbose:
            print("Starting iteration", iteration)
        factors[1] = factors[1]*T.reshape(weights, (1, -1))
        weights = T.ones(weights.shape, **tl.context(tensor_slices[0]))

        projections = _compute_projections(tensor_slices, factors, svd_fun, out=projections)
        projected_tensor = _project_tensor_slices(tensor_slices, projections, out=projected_tensor)
        _, factors = parafac(projected_tensor, rank, n_iter_max=n_iter_parafac, init=(weights, factors),
                             svd=svd, orthogonalise=False, verbose=verbose, return_errors=False,
                             normalize_factors=False, mask=None, random_state=random_state, tol=1e-100)

        if normalize_factors:
            new_factors = []
            for factor in factors:
                norms = T.norm(factor, axis=0)
                norms = tl.where(tl.abs(norms) <= tl.eps(factor.dtype), 
                    tl.ones(tl.shape(norms), **tl.context(factors[0])),
                    norms)

                weights = weights*norms
                new_factors.append(factor/(tl.reshape(norms, (1, -1))))

            factors = new_factors

            

        if tol:
            rec_error = _parafac2_reconstruction_error(tensor_slices, (weights, factors, projections))
            rec_error /= norm_tensor
            rec_errors.append(rec_error)

            if iteration >= 1:
                if verbose:
                    print('PARAFAC2 reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    if verbose:
                        print('converged in {} iterations.'.format(iteration))
                    break       
            else:
                if verbose:
                    print('PARAFAC2 reconstruction error={}'.format(rec_errors[-1]))

    parafac2_tensor = Parafac2Tensor((weights, factors, projections))

    if return_errors:
        return parafac2_tensor, rec_errors
    else:
        return parafac2_tensor
