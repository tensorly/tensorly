import warnings
import tensorly as tl
from tensorly.random import random_parafac2
from tensorly import backend as T
from . import parafac
from ..parafac2_tensor import parafac2_to_slice, Parafac2Tensor, _validate_parafac2_tensor
from ..kruskal_tensor import kruskal_normalise

# Authors: Marie Roald
#          Yngve Mardal Moe


def initialize_factors(tensor_slices, rank, random_state, non_negative):
    """Initiate a random PARAFAC2 decomposition given rank and tensor.
    """
    shapes = [m.shape for m in tensor_slices]
    return random_parafac2(shapes, rank, full=False, random_state=random_state)


def _compute_projections(tensor_slices, factors, svd_fun, out=None):
    A, B, C = factors

    if out is None:
        out = [T.zeros((tensor_slice.shape[0], C.shape[1]), **T.context(tensor_slice)) for tensor_slice in tensor_slices]

    for projection, a_i, tensor_slice in zip(out, A, tensor_slices):
        U, S, Vh = svd_fun(B@(a_i*C).T@tensor_slice.T, n_eigenvecs=A.shape[1])
        projection[...] = Vh.T@U.T
    
    return out


def _project_tensor_slices(tensor_slices, projections, out=None):
    if out is None:
        rank = projections[0].shape[1]
        num_slices = len(tensor_slices)
        num_cols = tensor_slices[0].shape[1]
        out = T.zeros((num_slices, rank, num_cols), **T.context(tensor_slices[0]))

    for projected_tensor_slice, tensor_slice, projection in zip(out, tensor_slices, projections):
        projected_tensor_slice[...] = T.dot(projection.T, tensor_slice)
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
             tol=1e-8, random_state=None, verbose=False, return_errors=False, non_negative=False,
             mask=None, n_iter_parafac=5):
    epsilon = 10e-12

    weights, factors, projections = initialize_factors(tensor_slices, rank, random_state=random_state,
                                                       non_negative=non_negative)

    rec_errors = []
    norm_tensor = tl.sqrt(sum(tl.norm(tensor_slice, 2) for tensor_slice in tensor_slices))
    svd_fun = _get_svd(svd)

    projected_tensor = tl.zeros([factor.shape[0] for factor in factors])

    for iteration in range(n_iter_max):
        if verbose:
            print("Starting iteration", iteration)
        factors[1] *= weights.reshape(1, -1)
        weights = T.ones(weights.shape, **tl.context(tensor_slices[0]))

        projections = _compute_projections(tensor_slices, factors, svd_fun, out=projections)
        projected_tensor = _project_tensor_slices(tensor_slices, projections, out=projected_tensor)
        _, factors = parafac(projected_tensor, rank, n_iter_max=n_iter_parafac, init=(weights, factors),
                             svd=svd, orthogonalise=False, verbose=verbose, return_errors=False,
                             normalize_factors=False, mask=None)
        # There are some issues with normalize_factors=True...
        for factor in factors:
            norms = T.norm(factor, axis=0)
            weights *= norms
            factor /= norms
        

        if tol:
            rec_error = _parafac2_reconstruction_error(tensor_slices, (weights, factors, projections))
            rec_errors.append(rec_error)

            if iteration >= 1:
                if verbose:
                    #print('reconstruction error={}, variation={}.'.format(
                    #    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                    print('PARAFAC2 reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    if verbose:
                        print('converged in {} iterations.'.format(iteration))
                    break       
            else:
                if verbose:
                    #print('reconstruction error={}'.format(rec_errors[-1]))
                    print('PARAFAC2 reconstruction error={}'.format(rec_errors[-1]))

    parafac2_tensor = Parafac2Tensor((weights, factors, projections))

    if return_errors:
        return parafac2_tensor, rec_errors
    else:
        return parafac2_tensor
