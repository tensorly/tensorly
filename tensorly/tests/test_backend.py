import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

from .. import backend as T
from ..base import fold, unfold
from ..base import partial_fold, partial_unfold
from ..base import tensor_to_vec, vec_to_tensor
from ..base import partial_tensor_to_vec, partial_vec_to_tensor

# Author: Jean Kossaifi

def test_unfold():
    """Test for unfold

    1. We do an exact test.
    
    2. Second,  a test inspired by the example in Kolda's paper:
       Even though we use a different definition of the unfolding,
       it should only differ by the ordering of the columns
    """
    X = T.tensor([[[1, 13],
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

    X = T.tensor(T.arange(24).reshape((3, 4, 2)))
    unfoldings = [T.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                            [8, 9, 10, 11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20, 21, 22, 23]]),
                  T.tensor([[0, 1, 8, 9, 16, 17],
                            [2, 3, 10, 11, 18, 19],
                            [4, 5, 12, 13, 20, 21],
                            [6, 7, 14, 15, 22, 23]]),
                  T.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])]
    for mode in range(X.ndim):
        unfolding = unfold(X, mode=mode)
        T.assert_array_equal(unfolding, unfoldings[mode])
        T.assert_array_equal(unfolding.reshape((-1, )), unfoldings[mode].reshape((-1,)))


def test_fold():
    """Test for fold
    """
    X = T.arange(24).reshape((3, 4, 2))
    unfoldings = [T.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                            [8, 9, 10, 11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20, 21, 22, 23]]),
    		  T.tensor([[0, 1, 8, 9, 16, 17],
                            [2, 3, 10, 11, 18, 19],
                            [4, 5, 12, 13, 20, 21],
                            [6, 7, 14, 15, 22, 23]]),
    		  T.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])]
    # hard coded example
    for mode in range(X.ndim):
    	T.assert_array_equal(fold(unfoldings[mode], mode, X.shape), X)

    # check dims
    for i in range(X.ndim):
        T.assert_array_equal(X, fold(unfold(X, i), i, X.shape))

    # chain unfolding and folding
    X = T.tensor(np.random.random(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
    for i in range(X.ndim):
        T.assert_array_equal(X, fold(unfold(X, i), i, X.shape))

 
def test_tensor_to_vec():
    """Test for tensor_to_vec"""
    X = T.tensor([[[ 0,  1],
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
    true_res = T.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                         12, 13, 14, 15, 16,  17, 18, 19, 20, 21, 22, 23])
    T.assert_array_equal(tensor_to_vec(X), true_res)


def test_vec_to_tensor():
    """Test for tensor_to_vec"""
    X = T.tensor([[[ 0,  1],
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
    vec = T.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                    12, 13, 14, 15, 16,  17, 18, 19, 20, 21, 22, 23])
    T.assert_array_equal(X, vec_to_tensor(vec, X.shape))

    # Convert to vector and back to tensor
    X = T.tensor(np.random.random((3, 4, 5, 2)))
    vec = tensor_to_vec(X)
    reconstructed = vec_to_tensor(vec, X.shape)
    T.assert_array_equal(X, reconstructed)


def test_partial_unfold():
    """Test for partial_unfold

    Notes
    -----
    Assumes that the standard unfold is correct!
    """
    X = T.arange(24).reshape((3, 4, 2))
    n_samples = 3
    ###################################
    # Samples are the first dimension #
    ###################################
    tensor = T.tensor(np.concatenate([np.arange(24).reshape((1, 3, 4, 2))+i\
                                      for i in range(n_samples)]))
    t = T.tensor(X)
    # We created here a tensor with 3 samples, each sample being similar to X
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, i, skip_begin=1)
        unfolded_X = unfold(t, i)
        for j in range(n_samples):  # test for each sample
            T.assert_array_equal(unfolded[j], unfolded_X+j)
    # Test for raveled tensor
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_begin=1, ravel_tensors=True)
        unfolded_X = unfold(t, i).reshape((-1, ))
        for j in range(n_samples):  # test for each sample
            T.assert_array_equal(unfolded[j], unfolded_X + j)
    
    ##################################
    # Samples are the last dimension #
    ##################################
    tensor = T.tensor(np.concatenate([np.arange(24).reshape((3, 4, 2, 1))+i\
                                      for i in range(n_samples)], axis=-1))
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_end=1, skip_begin=0)
        unfolded_X = unfold(t, i)
        for j in range(n_samples):  # test for each sample
            T.assert_array_equal(unfolded.T[j].T, unfolded_X+j)
    
    # Test for raveled tensor
    for i in range(X.ndim):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_end=1, skip_begin=0, ravel_tensors=True)
        unfolded_X = unfold(t, i).reshape((-1, ))
        for j in range(n_samples):  # test for each sample
            T.assert_array_equal(unfolded.T[j], unfolded_X+j)

def test_partial_fold():
    """Test for partial_fold

    Assumes partial unfolding works and check that
    refolding partially folded tensors results in
    the original tensor.
    """
    X = T.arange(24).reshape((3, 4, 2))
    unfolded = T.tensor([[[ 0,  1,  2,  3,  4,  5,  6,  7],
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
        T.assert_array_equal(folded[i], X)

    shape = [3, 4, 5, 6]
    X = T.tensor(np.random.random(shape))
    for i in [0, 1]:
        for mode in range(len(shape)-1):
            unfolded = partial_unfold(X, mode=mode, skip_begin=i, skip_end=(1-i))
            refolded = partial_fold(unfolded, mode=mode, shape=shape, skip_begin=i, skip_end=(1-i))
            T.assert_array_equal(refolded, X)

    # Test for raveled_tensor=True
    for i in [0, 1]:
        for mode in range(len(shape)-1):
            unfolded = partial_unfold(X, mode=mode, skip_begin=i, skip_end=(1-i), ravel_tensors=True)
            refolded = partial_fold(unfolded, mode=mode, shape=shape, skip_begin=i, skip_end=(1-i))
            T.assert_array_equal(refolded, X)



def test_partial_tensor_to_vec():
    """Test for partial_tensor_to_vec """
    X = np.arange(24).reshape((3, 4, 2))
    n_samples = 3

    ###################################
    # Samples are the first dimension #
    ###################################
    tensor = T.tensor(np.concatenate([X[None, ...]+i for i in range(n_samples)]))
    #we created here a tensor with 3 samples, each sample being similar to X
    vectorised = partial_tensor_to_vec(tensor, skip_begin=1)
    vec_X = tensor_to_vec(T.tensor(X))
    for j in range(n_samples): # test for each sample
        T.assert_array_equal(vectorised[j], vec_X+j)

    ##################################
    # Samples are the last dimension #
    ##################################
    tensor = T.tensor(np.concatenate([X[..., None]+i for i in range(n_samples)], axis=-1))
    vectorised = partial_tensor_to_vec(tensor, skip_end=1, skip_begin=0)
    vec_X = tensor_to_vec(T.tensor(X))
    for j in range(n_samples): # test for each sample
        T.assert_array_equal(vectorised.T[j], vec_X+j)


def test_partial_vec_to_tensor():
    """Test for partial_vec_to_tensor
    """
    X = np.arange(24).reshape((3, 4, 2))

    vectorised = T.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                            15, 16, 17, 18, 19, 20, 21, 22, 23],
                           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20, 21, 22, 23, 24],
                           [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                            17, 18, 19, 20, 21, 22, 23, 24, 25]])
    folded = partial_vec_to_tensor(vectorised, (3, 3, 4, 2), skip_begin=1)
    for i in range(3):
        T.assert_array_equal(folded[i], X+i)

    shape = [3, 4, 5, 6]
    X = T.tensor(np.random.random(shape))
    for i in [0, 1]:
        vec = partial_tensor_to_vec(X, skip_begin=i, skip_end=(1-i))
        ten = partial_vec_to_tensor(vec, shape=shape, skip_begin=i, skip_end=(1-i))
        T.assert_array_equal(X, ten)


def test_partial_svd():
    """Test for partial_svd"""
    sizes = [(100, 100), (100, 5), (10, 10), (5, 100)]
    n_eigenvecs = [10, 4, 5, 4]

    # Compare with sparse SVD
    for s, n in zip(sizes, n_eigenvecs):
        matrix = np.random.random(s)
        fU, fS, fV = T.partial_svd(T.tensor(matrix), n_eigenvecs=n)
        U, S, V = svds(matrix, k=n, which='LM')
        U, S, V = U[:, ::-1], S[::-1], V[::-1, :]
        T.assert_array_almost_equal(np.abs(S), T.abs(fS))
        T.assert_array_almost_equal(np.abs(U), T.abs(fU))
        T.assert_array_almost_equal(np.abs(V), T.abs(fV))

    # Compare with standard SVD
    sizes = [(100, 100), (100, 5), (10, 10), (10, 4), (5, 100)]
    n_eigenvecs = [10, 4, 5, 4, 4]
    for s, n in zip(sizes, n_eigenvecs):
        matrix = np.random.random(s)
        fU, fS, fV = T.partial_svd(T.tensor(matrix), n_eigenvecs=n)

        U, S, V = svd(matrix)
        U, S, V = U[:, :n], S[:n], V[:n, :]
        # Test for SVD
        T.assert_array_almost_equal(np.abs(S), T.abs(fS))
        T.assert_array_almost_equal(np.abs(U), T.abs(fU))
        T.assert_array_almost_equal(np.abs(V), T.abs(fV))

    with T.assert_raises(ValueError):
        tensor = T.tensor(np.random.random((3, 3, 3)))
        T.partial_svd(tensor)


def test_shape():
    A = T.arange(3*4*5)

    shape1 = (3*4,5)
    A1 = A.reshape(shape1)
    T.assert_equal(T.shape(A1), shape1)

    shape2 = (3,4,5)
    A2 = A.reshape(shape2)
    T.assert_equal(T.shape(A2), shape2)


def test_ndim():
    A = T.arange(3*4*5)
    T.assert_equal(T.ndim(A), 1)

    shape1 = (3*4,5)
    A1 = A.reshape(shape1)
    T.assert_equal(T.ndim(A1), 2)

    shape2 = (3,4,5)
    A2 = A.reshape(shape2)
    T.assert_equal(T.ndim(A2), 3)


def test_norm():
    v = T.tensor([1,2,3])
    T.assert_equal(T.norm(v,1), 6)

    A = T.arange(6).reshape((3,2))
    T.assert_equal(T.norm(A, 1), 15)

    column_norms1 = T.norm(A, 1, axis=0)
    row_norms1 = T.norm(A, 1, axis=1)
    T.assert_array_equal(column_norms1, T.tensor([6, 9]))
    T.assert_array_equal(row_norms1, T.tensor([1, 5, 9]))

    column_norms2 = T.norm(A, 2, axis=0)
    row_norms2 = T.norm(A, 2, axis=1)
    T.assert_array_almost_equal(column_norms2, T.tensor([4.47213602, 5.91608]))
    T.assert_array_almost_equal(row_norms2, T.tensor([1., 3.60555124, 6.40312433]))

    # limit as order->oo is the oo-norm
    column_norms10 = T.norm(A, 10, axis=0)
    row_norms10 = T.norm(A, 10, axis=1)
    T.assert_array_almost_equal(column_norms10, T.tensor([4.00039053, 5.00301552]))
    T.assert_array_almost_equal(row_norms10, T.tensor([1., 3.00516224, 5.05125666]))

