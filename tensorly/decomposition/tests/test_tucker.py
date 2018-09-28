import tensorly.backend as T
from tensorly.decomposition._tucker import tucker, partial_tucker, non_negative_tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import multi_mode_dot
from tensorly.random import check_random_state
from tensorly.testing import assert_equal, assert_


def test_partial_tucker():
    """Test for the Partial Tucker decomposition"""
    rng = check_random_state(1234)
    tol_norm_2 = 10e-3
    tol_max_abs = 10e-1
    tensor = T.tensor(rng.random_sample((3, 4, 3)))
    modes = [1, 2]
    core, factors = partial_tucker(tensor, modes, rank=None, n_iter_max=200, verbose=True)
    reconstructed_tensor = multi_mode_dot(core, factors, modes=modes)
    norm_rec = T.norm(reconstructed_tensor, 2)
    norm_tensor = T.norm(tensor, 2)
    assert_((norm_rec - norm_tensor)/norm_rec < tol_norm_2)

    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.max(T.abs(norm_rec - norm_tensor)) < tol_max_abs)

    # Test the shape of the core and factors
    ranks = [3, 1]
    core, factors = partial_tucker(tensor, modes=modes, rank=ranks, n_iter_max=100, verbose=1)
    for i, rank in enumerate(ranks):
        assert_equal(factors[i].shape, (tensor.shape[i+1], ranks[i]),
                     err_msg="factors[{}].shape={}, expected {}".format(
                         i, factors[i].shape, (tensor.shape[i+1], ranks[i])))
    assert_equal(core.shape, [tensor.shape[0]]+ranks, err_msg="Core.shape={}, "
                     "expected {}".format(core.shape, [tensor.shape[0]]+ranks))


def test_tucker():
    """Test for the Tucker decomposition"""
    rng = check_random_state(1234)

    tol_norm_2 = 10e-3
    tol_max_abs = 10e-1
    tensor = T.tensor(rng.random_sample((3, 4, 3)))
    core, factors = tucker(tensor, rank=None, n_iter_max=200, verbose=True)
    reconstructed_tensor = tucker_to_tensor(core, factors)
    norm_rec = T.norm(reconstructed_tensor, 2)
    norm_tensor = T.norm(tensor, 2)
    assert((norm_rec - norm_tensor)/norm_rec < tol_norm_2)

    # Test the max abs difference between the reconstruction and the tensor
    assert(T.max(T.abs(reconstructed_tensor - tensor)) < tol_max_abs)

    # Test the shape of the core and factors
    ranks = [2, 3, 1]
    core, factors = tucker(tensor, rank=ranks, n_iter_max=100, verbose=1)
    for i, rank in enumerate(ranks):
        assert_equal(factors[i].shape, (tensor.shape[i], ranks[i]),
                     err_msg="factors[{}].shape={}, expected {}".format(
                         i, factors[i].shape, (tensor.shape[i], ranks[i])))
        assert_equal(T.shape(core)[i], rank, err_msg="Core.shape[{}]={}, "
                     "expected {}".format(i, core.shape[i], rank))

    # Random and SVD init should converge to a similar solution
    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1

    core_svd, factors_svd = tucker(tensor, rank=[3, 4, 3], n_iter_max=200, init='svd', verbose=1)
    core_random, factors_random = tucker(tensor, rank=[3, 4, 3], n_iter_max=200, init='random', random_state=1234)
    rec_svd = tucker_to_tensor(core_svd, factors_svd)
    rec_random = tucker_to_tensor(core_random, factors_random)
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(T.max(T.abs(rec_svd - rec_random)) < tol_max_abs,
            'abs norm of difference between svd and random init too high')


def test_non_negative_tucker():
    """Test for non-negative Tucker"""
    rng = check_random_state(1234)

    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    tensor = T.tensor(rng.random_sample((3, 4, 3)) + 1)
    core, factors = tucker(tensor, rank=[3, 4, 3], n_iter_max=200, verbose=1)
    nn_core, nn_factors = non_negative_tucker(tensor, rank=[3, 4, 3], n_iter_max=100)

    # Make sure all components are positive
    for factor in nn_factors:
        assert_(T.all(factor >= 0))
    assert_(T.all(nn_core >= 0))

    reconstructed_tensor = tucker_to_tensor(core, factors)
    nn_reconstructed_tensor = tucker_to_tensor(nn_core, nn_factors)
    error = T.norm(reconstructed_tensor - nn_reconstructed_tensor, 2)
    error /= T.norm(reconstructed_tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction error higher than tol')

    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.norm(reconstructed_tensor - nn_reconstructed_tensor, 'inf') < tol_max_abs,
              'abs norm of reconstruction error higher than tol')

    core_svd, factors_svd = non_negative_tucker(tensor, rank=[3, 4, 3], n_iter_max=500, init='svd', verbose=1)
    core_random, factors_random = non_negative_tucker(tensor, rank=[3, 4, 3], n_iter_max=200, init='random', random_state=1234)
    rec_svd = tucker_to_tensor(core_svd, factors_svd)
    rec_random = tucker_to_tensor(core_random, factors_random)
    error = T.norm(rec_svd - rec_random, 2)
    error /= T.norm(rec_svd, 2)
    assert_(error < tol_norm_2,
            'norm 2 of difference between svd and random init too high')
    assert_(T.norm(rec_svd - rec_random, 'inf') < tol_max_abs,
            'abs norm of difference between svd and random init too high')
