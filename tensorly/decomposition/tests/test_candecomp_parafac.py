from numpy.testing import assert_, assert_array_almost_equal
import numpy as np
from ..candecomp_parafac import parafac
from ..candecomp_parafac import non_negative_parafac
from ...kruskal import kruskal_to_tensor
from ...tenalg import norm
from ...utils import check_random_state


def test_parafac():
    """Test for the CANDECOMP-PARAFAC decomposition
    """
    rng = check_random_state(1234)
    tol_norm_2 = 10e-2
    tol_max_abs = 10e-2
    tensor = rng.random_sample((3, 4, 2))
    factors_svd = parafac(tensor, rank=4, n_iter_max=200, init='svd')
    factors_random = parafac(tensor, rank=4, n_iter_max=200, init='random', verbose=1)
    rec_svd = kruskal_to_tensor(factors_svd)
    rec_random = kruskal_to_tensor(factors_random)
    error = norm(rec_svd - tensor, 2)
    error /= norm(tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(np.max(np.abs(rec_svd - tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    error = norm(rec_svd - rec_random, 2)
    error /= norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(np.max(np.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')



def test_non_negative_parafac():
    """Test for non-negative PARAFAC

    TODO: more rigorous test
    """
    tol_norm_2 = 10e-1
    tol_max_abs = 1
    rng = check_random_state(1234)
    tensor = rng.random_sample((3, 3, 3))*2
    factors = parafac(tensor, rank=3, n_iter_max=120)
    nn_factors = non_negative_parafac(tensor, rank=3, n_iter_max=500, init='svd', verbose=1)

    # Make sure all components are positive
    for factor in nn_factors:
        assert_(np.all(factor >= 0))

    reconstructed_tensor = kruskal_to_tensor(factors)
    nn_reconstructed_tensor = kruskal_to_tensor(nn_factors)
    error = norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    assert_(np.max(np.abs(reconstructed_tensor - nn_reconstructed_tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')

    factors_svd = non_negative_parafac(tensor, rank=3, n_iter_max=100,
                                       init='svd')
    factors_random = non_negative_parafac(tensor, rank=3, n_iter_max=100,
                                          init='random', verbose=1)
    rec_svd = kruskal_to_tensor(factors_svd)
    rec_random = kruskal_to_tensor(factors_random)
    error = norm(rec_svd - rec_random, 2)
    error /= norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(np.max(np.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')
