import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_, assert_raises
from ..base import tensor_from_frontal_slices, unfold, fold
from ..base import partial_unfold, partial_fold
from ..base import tensor_to_vec, vec_to_tensor
from ..base import partial_tensor_to_vec, partial_vec_to_tensor


def test_tensor_from_frontal_slices():
    """Test for tensor_from_frontal_slices"""
    X1 = np.array([[1, 4, 7, 10],
                   [2, 5, 8, 11],
                   [3, 6, 9, 12]])
    X2 = np.array([[13, 16, 19, 22],
                   [14, 17, 20, 23],
                   [15, 18, 21, 24]])
    res = tensor_from_frontal_slices(X1, X2)
    X = np.array([[[1, 13],
                   [4, 16],
                   [7, 19],
                   [10, 22]],

                  [[2, 14],
                   [5, 17],
                   [8, 20],
                   [11, 23]],

                  [[3, 15],
                   [6, 18],
                   [9, 21],
                   [12, 24]]])
    assert_array_equal(res, X)


def test_unfold():
    """Test for unfold

    1. First a test inspired by the example in Kolda's paper:
       Even though we use a slightly different unfolding, as the
       order of the columns is not important but they should all be there

    2. Second we do an exact test.
    """
    X = np.array([[[1, 13],
                   [4, 16],
                   [7, 19],
                   [10, 22]],

                  [[2, 14],
                   [5, 17],
                   [8, 20],
                   [11, 23]],

                  [[3, 15],
                   [6, 18],
                   [9, 21],
                   [12, 24]]])

    unfolded_mode_0 = np.array([[1, 4, 7, 10, 13, 16, 19, 22],
                                [2, 5, 8, 11, 14, 17, 20, 23],
                                [3, 6, 9, 12, 15, 18, 21, 24]])
    unfolded_mode_1 = np.array([[1, 2, 3, 13, 14, 15],
                                [4, 5, 6, 16, 17, 18],
                                [7, 8, 9, 19, 20, 21],
                                [10, 11, 12, 22, 23, 24]])
    unfolded_mode_2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])
    true_unfoldings = [unfolded_mode_0, unfolded_mode_1, unfolded_mode_2]

    for mode in range(X.ndim):
        unfolding = unfold(X, mode)
        for column in true_unfoldings[mode].T:
            assert_(column in unfolding.T)

    # Now an exact test
    X = np.arange(24).reshape((3, 4, 2))
    unfolded_mode_0 = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                [8, 9, 10, 11, 12, 13, 14, 15],
                                [16, 17, 18, 19, 20, 21, 22, 23]])
    unfolded_mode_1 = np.array([[0, 1, 8, 9, 16, 17],
                                [2, 3, 10, 11, 18, 19],
                                [4, 5, 12, 13, 20, 21],
                                [6, 7, 14, 15, 22, 23]])
    unfolded_mode_2 = np.array([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])

    assert_array_equal(unfold(X, mode=0), unfolded_mode_0)
    assert_array_equal(unfold(X, mode=0).ravel(), unfolded_mode_0.ravel())
    assert_array_equal(unfold(X, mode=1), unfolded_mode_1)
    assert_array_equal(unfold(X, mode=1).ravel(), unfolded_mode_1.ravel())
    assert_array_equal(unfold(X, mode=2), unfolded_mode_2)
    assert_array_equal(unfold(X, mode=2).ravel(), unfolded_mode_2.ravel())


def test_fold():
    """Test for fold
    """
    X = np.arange(24).reshape((3, 4, 2))
    unfolded_mode_0 = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                [8, 9, 10, 11, 12, 13, 14, 15],
                                [16, 17, 18, 19, 20, 21, 22, 23]])
    unfolded_mode_1 = np.array([[0, 1, 8, 9, 16, 17],
                                [2, 3, 10, 11, 18, 19],
                                [4, 5, 12, 13, 20, 21],
                                [6, 7, 14, 15, 22, 23]])
    unfolded_mode_2 = np.array([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])
    # hard coded example
    assert_array_equal(fold(unfolded_mode_0, 0, X.shape), X)
    assert_array_equal(fold(unfolded_mode_1, 1, X.shape), X)
    assert_array_equal(fold(unfolded_mode_2, 2, X.shape), X)

    # check dims
    for i in range(X.ndim):
        assert_array_equal(X, fold(unfold(X, i), i, X.shape))

    # chain unfolding and folding
    X = np.random.random(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    for i in range(X.ndim):
        assert_array_equal(X, fold(unfold(X, i), i, X.shape))


def test_tensor_to_vec():
    """Test for tensor_to_vec"""
    X = np.array([[[ 0,  1],
                   [ 2,  3],
                   [ 4,  5],
                   [ 6,  7]],

                  [[ 8,  9],
                   [10, 11],
                   [12, 13],
                   [14, 15]],

                  [[16, 17],
                   [18, 19],
                   [20, 21],
                   [22, 23]]])
    true_res = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                         12, 13, 14, 15, 16,  17, 18, 19, 20, 21, 22, 23])
    assert_array_equal(tensor_to_vec(X), true_res)


def test_vec_to_tensor():
    """Test for tensor_to_vec"""
    X = np.array([[[ 0,  1],
                   [ 2,  3],
                   [ 4,  5],
                   [ 6,  7]],

                  [[ 8,  9],
                   [10, 11],
                   [12, 13],
                   [14, 15]],

                  [[16, 17],
                   [18, 19],
                   [20, 21],
                   [22, 23]]])
    vec = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                    12, 13, 14, 15, 16,  17, 18, 19, 20, 21, 22, 23])
    assert_array_equal(X, vec_to_tensor(vec, X.shape))

    # Convert to vector and back to tensor
    X = np.random.random((3, 4, 5, 2))
    vec = tensor_to_vec(X)
    reconstructed = vec_to_tensor(vec, X.shape)
    assert_array_equal(X, reconstructed)


def test_partial_unfold():
    """Test for partial_unfold

    Notes
    -----
    Assumes that the standard unfold is correct!
    """
    X = np.array([[[1, 13],
                   [4, 16],
                   [7, 19],
                   [10, 22]],

                  [[2, 14],
                   [5, 17],
                   [8, 20],
                   [11, 23]],

                  [[3, 15],
                   [6, 18],
                   [9, 21],
                   [12, 24]]])
    n_samples = 3

    ###################################
    # Samples are the first dimension #
    ###################################
    tensor = np.concatenate([X[None, ...]+i for i in range(n_samples)])
    # We created here a tensor with 3 samples, each sample being similar to X
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, i, skip_begin=1)
        unfolded_X = unfold(X, i)
        for j in range(n_samples):  # test for each sample
            assert_array_equal(unfolded[j, ...], unfolded_X+j)
    # Test for raveled tensor
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_begin=1, ravel_tensors=True)
        unfolded_X = unfold(X, i).ravel()
        for j in range(n_samples):  # test for each sample
            assert_array_equal(unfolded[j, ...], unfolded_X + j)

    ##################################
    # Samples are the last dimension #
    ##################################
    tensor = np.concatenate([X[..., None]+i for i in range(n_samples)], axis=-1)
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_end=1, skip_begin=0)
        unfolded_X = unfold(X, i)
        for j in range(n_samples):  # test for each sample
            assert_array_equal(unfolded[..., j], unfolded_X+j)

    # Test for raveled tensor
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_end=1, skip_begin=0, ravel_tensors=True)
        unfolded_X = unfold(X, i).ravel()
        for j in range(n_samples):  # test for each sample
            assert_array_equal(unfolded[..., j], unfolded_X+j)



def test_partial_fold():
    """Test for partial_fold

    Assumes partial unfolding works and check that
    refolding partially folded tensors results in
    the original tensor.
    """
    X = np.arange(24).reshape((3, 4, 2))
    unfolded = np.array([[[ 0,  1,  2,  3,  4,  5,  6,  7],
                          [ 8,  9, 10, 11, 12, 13, 14, 15],
                          [16, 17, 18, 19, 20, 21, 22, 23]],
                         [[ 0,  1,  2,  3,  4,  5,  6,  7],
                          [ 8,  9, 10, 11, 12, 13, 14, 15],
                          [16, 17, 18, 19, 20, 21, 22, 23]],
                         [[ 0,  1,  2,  3,  4,  5,  6,  7],
                          [ 8,  9, 10, 11, 12, 13, 14, 15],
                          [16, 17, 18, 19, 20, 21, 22, 23]]])
    folded = partial_fold(unfolded, 0, (3, 3, 4, 2), skip_begin=1)
    for i in range(3):
        assert_array_equal(folded[i, ...], X)

    shape = [3, 4, 5, 6]
    X = np.random.random(shape)
    for i in [0, 1]:
        for mode in range(len(shape)-1):
            unfolded = partial_unfold(X, mode=mode, skip_begin=i, skip_end=(1-i))
            refolded = partial_fold(unfolded, mode=mode, shape=shape, skip_begin=i, skip_end=(1-i))
            assert_array_equal(refolded, X)

    # Test for raveled_tensor=True
    for i in [0, 1]:
        for mode in range(len(shape)-1):
            unfolded = partial_unfold(X, mode=mode, skip_begin=i, skip_end=(1-i), ravel_tensors=True)
            refolded = partial_fold(unfolded, mode=mode, shape=shape, skip_begin=i, skip_end=(1-i))
            assert_array_equal(refolded, X)



def test_partial_tensor_to_vec():
    """Test for partial_tensor_to_vec """
    X = np.arange(24).reshape((3, 4, 2))
    n_samples = 3

    ###################################
    # Samples are the first dimension #
    ###################################
    tensor = np.concatenate([X[None, ...]+i for i in range(n_samples)])
    #we created here a tensor with 3 samples, each sample being similar to X
    vectorised = partial_tensor_to_vec(tensor, skip_begin=1)
    vec_X = tensor_to_vec(X)
    for j in range(n_samples): # test for each sample
        assert_array_equal(vectorised[j, ...], vec_X+j)

    ##################################
    # Samples are the last dimension #
    ##################################
    tensor = np.concatenate([X[..., None]+i for i in range(n_samples)], axis=-1)
    vectorised = partial_tensor_to_vec(tensor, skip_end=1, skip_begin=0)
    vec_X = tensor_to_vec(X)
    for j in range(n_samples): # test for each sample
        assert_array_equal(vectorised[..., j], vec_X+j)


def test_partial_vec_to_tensor():
    """Test for partial_vec_to_tensor
    """
    X = np.arange(24).reshape((3, 4, 2))

    vectorised = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                            15, 16, 17, 18, 19, 20, 21, 22, 23],
                           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20, 21, 22, 23, 24],
                           [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                            17, 18, 19, 20, 21, 22, 23, 24, 25]])
    folded = partial_vec_to_tensor(vectorised, (3, 3, 4, 2), skip_begin=1)
    for i in range(3):
        assert_array_equal(folded[i, ...], X+i)

    shape = [3, 4, 5, 6]
    X = np.random.random(shape)
    for i in [0, 1]:
        vec = partial_tensor_to_vec(X, skip_begin=i, skip_end=(1-i))
        ten = partial_vec_to_tensor(vec, shape=shape, skip_begin=i, skip_end=(1-i))
        assert_array_equal(X, ten)