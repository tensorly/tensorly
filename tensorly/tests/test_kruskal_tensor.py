import numpy as np

import tensorly as tl
from ..tenalg import khatri_rao, mode_dot
from ..kruskal_tensor import (kruskal_to_tensor, kruskal_to_unfolded, 
                              kruskal_to_vec, _validate_kruskal_tensor,
                              kruskal_normalise, KruskalTensor,
                              kruskal_mode_dot, unfolding_dot_khatri_rao,
                              kruskal_norm)
from ..base import unfold, tensor_to_vec
from tensorly.random import check_random_state, random_kruskal
from tensorly.testing import (assert_equal, assert_raises, assert_,
                              assert_array_equal, assert_array_almost_equal)


def test_kruskal_normalise():
    shape = (3, 4, 5)
    rank = 4
    kruskal_tensor = random_kruskal(shape, rank)
    weights, factors = kruskal_normalise(kruskal_tensor)
    expected_norm = tl.ones(rank)
    for f in factors:
        assert_array_almost_equal(tl.norm(f, axis=0), expected_norm)
    assert_array_almost_equal(kruskal_to_tensor((weights, factors)), kruskal_to_tensor(kruskal_tensor))
    

def test_validate_kruskal_tensor():
    rng = check_random_state(12345)
    true_shape = (3, 4, 5)
    true_rank = 3
    kruskal_tensor = random_kruskal(true_shape, true_rank)
    (weights, factors) = kruskal_normalise(kruskal_tensor)
    
    # Check correct rank and shapes are returned
    shape, rank = _validate_kruskal_tensor((weights, factors))
    assert_equal(shape, true_shape,
                    err_msg='Returned incorrect shape (got {}, expected {})'.format(
                        shape, true_shape))
    assert_equal(rank, true_rank,
                    err_msg='Returned incorrect rank (got {}, expected {})'.format(
                        rank, true_rank))
    
    # One of the factors has the wrong rank
    factors[0], copy = tl.tensor(rng.random_sample((4, 4))), factors[0]
    with assert_raises(ValueError):
        _validate_kruskal_tensor((weights, factors))
    
    # Not the correct amount of weights
    factors[0] = copy
    wrong_weights = weights[1:]
    with assert_raises(ValueError):
        _validate_kruskal_tensor((wrong_weights, factors))

    # Not enough factors
    with assert_raises(ValueError):
        _validate_kruskal_tensor((weights[:1], factors[:1]))

def test_kruskal_to_tensor():
    """Test for kruskal_to_tensor."""
    U1 = np.reshape(np.arange(1, 10), (3, 3))
    U2 = np.reshape(np.arange(10, 22), (4, 3))
    U3 = np.reshape(np.arange(22, 28), (2, 3))
    U4 = np.reshape(np.arange(28, 34), (2, 3))
    U = [tl.tensor(t) for t in [U1, U2, U3, U4]]
    true_res = tl.tensor([[[[  46754.,   51524.],
                            [  52748.,   58130.]],

                           [[  59084.,   65114.],
                            [  66662.,   73466.]],

                           [[  71414.,   78704.],
                            [  80576.,   88802.]],

                           [[  83744.,   92294.],
                            [  94490.,  104138.]]],


                          [[[ 113165.,  124784.],
                            [ 127790.,  140912.]],

                           [[ 143522.,  158264.],
                            [ 162080.,  178730.]],

                           [[ 173879.,  191744.],
                            [ 196370.,  216548.]],

                           [[ 204236.,  225224.],
                            [ 230660.,  254366.]]],


                          [[[ 179576.,  198044.],
                            [ 202832.,  223694.]],

                           [[ 227960.,  251414.],
                            [ 257498.,  283994.]],

                           [[ 276344.,  304784.],
                            [ 312164.,  344294.]],

                           [[ 324728.,  358154.],
                            [ 366830.,  404594.]]]])
    res = kruskal_to_tensor((tl.ones(3), U))
    assert_array_equal(res, true_res, err_msg='Khatri-rao incorrectly transformed into full tensor.')

    columns = 4
    rows = [3, 4, 2]
    matrices = [tl.tensor(np.arange(k * columns).reshape((k, columns))) for k in rows]
    tensor = kruskal_to_tensor((tl.ones(columns), matrices))
    for i in range(len(rows)):
        unfolded = unfold(tensor, mode=i)
        U_i = matrices.pop(i)
        reconstructed = tl.dot(U_i, tl.transpose(khatri_rao(matrices)))
        assert_array_almost_equal(reconstructed, unfolded)
        matrices.insert(i, U_i)

def test_kruskal_to_tensor_with_weights():
    A = tl.reshape(tl.arange(1,5), (2,2))
    B = tl.reshape(tl.arange(5,9), (2,2))
    weigths = tl.tensor([2,-1])

    out = kruskal_to_tensor((weigths, [A,B]))
    expected = tl.tensor([[-2,-2], [6, 10]])  # computed by hand
    assert_array_equal(out, expected)

    (weigths, factors) = random_kruskal((5, 5, 5), rank=5, normalise_factors=True, full=False)
    true_res = tl.dot(tl.dot(factors[0], tl.diag(weigths)),
                      tl.transpose(tl.tenalg.khatri_rao(factors[1:])))
    true_res = tl.fold(true_res, 0, (5, 5, 5))  
    res = kruskal_to_tensor((weigths, factors))
    assert_array_almost_equal(true_res, res,
     err_msg='weights incorrectly incorporated in kruskal_to_tensor')

def test_kruskal_to_unfolded():
    """Test for kruskal_to_unfolded.
        !!Assumes that kruskal_to_tensor and unfold are properly tested and work!!
    """
    U1 = np.reshape(np.arange(1, 10), (3, 3))
    U2 = np.reshape(np.arange(10, 22), (4, 3))
    U3 = np.reshape(np.arange(22, 28), (2, 3))
    U4 = np.reshape(np.arange(28, 34), (2, 3))
    U = [tl.tensor(t) for t in [U1, U2, U3, U4]]
    kruskal_tensor = KruskalTensor((tl.ones(3), U))

    full_tensor = kruskal_to_tensor(kruskal_tensor)
    for mode in range(4):
        true_res = unfold(full_tensor, mode)
        res = kruskal_to_unfolded(kruskal_tensor, mode)
        assert_array_equal(true_res, res, err_msg='khatri_rao product unfolded incorrectly for mode {}.'.format(mode))

def test_kruskal_to_vec():
    """Test for kruskal_to_vec"""
    U1 = np.reshape(np.arange(1, 10), (3, 3))
    U2 = np.reshape(np.arange(10, 22), (4, 3))
    U3 = np.reshape(np.arange(22, 28), (2, 3))
    U4 = np.reshape(np.arange(28, 34), (2, 3))
    U = [tl.tensor(t) for t in [U1, U2, U3, U4]]
    kruskal_tensor = KruskalTensor((tl.ones(3), U))
    full_tensor = kruskal_to_tensor(kruskal_tensor)
    true_res = tensor_to_vec(full_tensor)
    res = kruskal_to_vec(kruskal_tensor)
    assert_array_equal(true_res, res, err_msg='khatri_rao product converted incorrectly to vec.')

def test_kruskal_mode_dot():
    """Test for kruskal_mode_dot
    
        We will compare kruskal_mode_dot 
        (which operates directly on decomposed tensors)
        with mode_dot (which operates on full tensors)
        and check that the results are the same.
    """
    rng = check_random_state(12345)
    shape = (5, 4, 6)
    rank = 3
    kruskal_ten = random_kruskal(shape, rank=rank, orthogonal=True, full=False)
    full_tensor = tl.kruskal_to_tensor(kruskal_ten)
    # matrix for mode 1
    matrix = tl.tensor(rng.random_sample((7, shape[1])))
    # vec for mode 2
    vec = tl.tensor(rng.random_sample(shape[2]))

    # Test kruskal_mode_dot with matrix
    res = kruskal_mode_dot(kruskal_ten, matrix, mode=1, copy=True)
    # Note that if copy=True is not respected, factors will be changes
    # And the next test will fail
    res = tl.kruskal_to_tensor(res)
    true_res = mode_dot(full_tensor, matrix, mode=1)
    assert_array_almost_equal(true_res, res)
    
    # Check that the data was indeed copied
    rec = tl.kruskal_to_tensor(kruskal_ten)
    assert_array_almost_equal(full_tensor, rec)
    
    # Test kruskal_mode_dot with vec
    res = kruskal_mode_dot(kruskal_ten, vec, mode=2, copy=True)
    res = tl.kruskal_to_tensor(res)
    true_res = mode_dot(full_tensor, vec, mode=2)
    assert_equal(res.shape, true_res.shape)
    assert_array_almost_equal(true_res, res)


def test_unfolding_dot_khatri_rao():
    """Test for unfolding_dot_khatri_rao
    
    Check against other version check sparse safe
    """
    shape = (10, 10, 10, 4)
    rank = 5
    tensor = tl.tensor(np.random.random(shape))
    weights, factors = random_kruskal(shape=shape, rank=rank, 
                                      full=False, normalise_factors=True)
    
    for mode in range(tl.ndim(tensor)):
        # Version forming explicitely the khatri-rao product
        unfolded = unfold(tensor, mode)
        kr_factors = khatri_rao(factors, weights=weights, skip_matrix=mode)
        true_res = tl.dot(unfolded, kr_factors)

        # Efficient sparse-safe version
        res = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)
        assert_array_almost_equal(true_res, res, decimal=3)


def test_kruskal_norm():
    """Test for kruskal_norm
    """
    shape = (8, 5, 6, 4)
    rank = 25
    kruskal_tensor = random_kruskal(shape=shape, rank=rank, 
                                      full=False, normalise_factors=True)
    tol = 10e-5
    rec = tl.kruskal_to_tensor(kruskal_tensor)
    true_res = tl.norm(rec, 2)
    res = kruskal_norm(kruskal_tensor)
    assert_(tl.abs(true_res - res) <= tol)
