import numpy as np
from ..candecomp_parafac import parafac
from ..candecomp_parafac import non_negative_parafac
from ...kruskal_tensor import kruskal_to_tensor
from ...random import check_random_state
from ... import backend as T

def test_parafac():
    """Test for the CANDECOMP-PARAFAC decomposition
    """
    rng = check_random_state(1234)
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor = T.tensor(rng.random_sample((3, 4, 2)))
    factors_svd = parafac(tensor, rank=4, n_iter_max=200, init='svd', tol=10e-5)
    factors_random = parafac(tensor, rank=4, n_iter_max=200, init='random', tol=10e-5, random_state=1234, verbose=0)
    rec_svd = kruskal_to_tensor(factors_svd)
    rec_random = kruskal_to_tensor(factors_random)
    error = T.norm(rec_svd - tensor, 2)
    error /= T.norm(tensor, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    T.assert_(T.max(T.abs(rec_svd - tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    T.assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')


def test_parafac_with_eigenvalues():
    """Test for the CANDECOMP-PARAFAC decomposition with eigenvalues
    """
    rng = check_random_state(1234)
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor = T.tensor(rng.random_sample((3, 4, 2)))
    factors_svd, eigenvalues_svd = parafac(
        tensor, rank=4, n_iter_max=200, init='svd', tol=10e-5,
        with_eigenvalues=True)
    factors_random, eigenvalues_random = parafac(
        tensor, rank=4, n_iter_max=200, init='random', tol=10e-5,
        random_state=1234, with_eigenvalues=True, verbose=0)
    rec_svd = kruskal_to_tensor(factors_svd, eigenvalues=eigenvalues_svd)
    rec_random = kruskal_to_tensor(factors_random, eigenvalues=eigenvalues_random)
    error = T.norm(rec_svd - tensor, 2)
    error /= T.norm(tensor, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    T.assert_(T.max(T.abs(rec_svd - tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    T.assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')


def test_non_negative_parafac():
    """Test for non-negative PARAFAC

    TODO: more rigorous test
    """
    tol_norm_2 = 10e-1
    tol_max_abs = 1
    rng = check_random_state(1234)
    tensor = T.tensor(rng.random_sample((3, 3, 3))+1)
    factors = parafac(tensor, rank=3, n_iter_max=120)
    nn_factors = non_negative_parafac(tensor, rank=3, n_iter_max=100, tol=10e-4, init='svd', verbose=0)

    # Make sure all components are positive
    for factor in nn_factors:
        T.assert_(T.all(factor >= 0))

    reconstructed_tensor = kruskal_to_tensor(factors)
    nn_reconstructed_tensor = kruskal_to_tensor(nn_factors)
    error = T.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= T.norm(reconstructed_tensor, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    T.assert_(T.max(T.abs(reconstructed_tensor - nn_reconstructed_tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    factors_svd = non_negative_parafac(tensor, rank=3, n_iter_max=100, tol=10e-4,
                                       init='svd')
    factors_random = non_negative_parafac(tensor, rank=3, n_iter_max=100, tol=10e-4,
                                          init='random', random_state=1234, verbose=0)
    rec_svd = kruskal_to_tensor(factors_svd)
    rec_random = kruskal_to_tensor(factors_random)
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    T.assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    T.assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')

