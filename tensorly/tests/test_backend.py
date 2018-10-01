from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np
from scipy.linalg import svd

import tensorly as tl
from tensorly import backend as T
from tensorly.base import (fold, unfold, partial_fold, partial_unfold,
                           tensor_to_vec, vec_to_tensor, partial_tensor_to_vec,
                           partial_vec_to_tensor)
from tensorly.testing import (assert_array_equal, assert_equal, assert_,
                              assert_array_almost_equal, assert_raises)

# Author: Jean Kossaifi


def test_set_backend():
    torch = pytest.importorskip('torch')

    toplevel_backend = tl.get_backend()

    # Set in context manager
    with tl.set_backend('numpy'):
        assert tl.get_backend() == 'numpy'
        assert isinstance(tl.tensor([1, 2, 3]), np.ndarray)
        assert isinstance(T.tensor([1, 2, 3]), np.ndarray)
        assert tl.float32 is T.float32 is np.float32

        with tl.set_backend('pytorch'):
            assert tl.get_backend() == 'pytorch'
            assert torch.is_tensor(tl.tensor([1, 2, 3]))
            assert torch.is_tensor(T.tensor([1, 2, 3]))
            assert tl.float32 is T.float32 is torch.float32

        # Sets back to numpy
        assert tl.get_backend() == 'numpy'
        assert isinstance(tl.tensor([1, 2, 3]), np.ndarray)
        assert isinstance(T.tensor([1, 2, 3]), np.ndarray)
        assert tl.float32 is T.float32 is np.float32

    # Reset back to initial backend
    assert tl.get_backend() == toplevel_backend

    # Set not in context manager
    try:
        obj = tl.set_backend('pytorch')
        assert tl.get_backend() == 'pytorch'
        assert repr(obj) == "tensorly.set_backend('pytorch')"
    finally:
        tl.set_backend(toplevel_backend)

    assert tl.get_backend() == toplevel_backend

    # Improper name doesn't reset backend
    with assert_raises(ValueError):
        tl.set_backend('not-a-real-backend')
    assert tl.get_backend() == toplevel_backend


def test_set_backend_make_default():
    pytest.importorskip('torch')

    global_default = tl.get_backend()

    with ThreadPoolExecutor(max_workers=1) as executor:

        with tl.set_backend('numpy'):
            assert tl.get_backend() == 'numpy'
            # Changes only happen locally in this thread
            assert executor.submit(tl.get_backend).result() == global_default

        # Set the global default backend
        try:
            tl.set_backend('pytorch', make_default=True)

            # Changed toplevel default in all threads
            assert executor.submit(tl.get_backend).result() == 'pytorch'

            with tl.set_backend('numpy'):
                assert tl.get_backend() == 'numpy'

                def check():
                    assert tl.get_backend() == 'pytorch'
                    with tl.set_backend('numpy') as ctx:
                        assert tl.get_backend() == 'numpy'
                    assert tl.get_backend() == 'pytorch'

                executor.submit(check).result()
        finally:
            tl.set_backend(global_default, make_default=True)

        assert tl.get_backend() == global_default
        assert executor.submit(tl.get_backend).result() == global_default


def test_backend_and_tensorly_module_attributes():
    for dtype in ['int32', 'int64', 'float32', 'float64']:
        assert dtype in dir(tl)
        assert dtype in dir(T)
        assert getattr(T, dtype) is getattr(tl, dtype)

    with assert_raises(AttributeError):
        tl.not_a_real_attribute


def test_tensor_creation():
    tensor = T.tensor(np.arange(12).reshape((4, 3)))
    tensor2 = tl.tensor(np.arange(12).reshape((4, 3)))

    assert T.is_tensor(tensor)
    assert T.is_tensor(tensor2)


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

    X = T.reshape(T.arange(24), (3, 4, 2))
    unfoldings = [T.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                            [8, 9, 10, 11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20, 21, 22, 23]]),
                  T.tensor([[0, 1, 8, 9, 16, 17],
                            [2, 3, 10, 11, 18, 19],
                            [4, 5, 12, 13, 20, 21],
                            [6, 7, 14, 15, 22, 23]]),
                  T.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])]
    for mode in range(T.ndim(X)):
        unfolding = unfold(X, mode=mode)
        assert_array_equal(unfolding, unfoldings[mode])
        assert_array_equal(T.reshape(unfolding, (-1, )),
                           T.reshape(unfoldings[mode], (-1,)))


def test_fold():
    """Test for fold
    """
    X = T.reshape(T.arange(24), (3, 4, 2))
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
    for mode in range(T.ndim(X)):
        assert_array_equal(fold(unfoldings[mode], mode, X.shape), X)

    # check dims
    for i in range(T.ndim(X)):
        assert_array_equal(X, fold(unfold(X, i), i, X.shape))

    # chain unfolding and folding
    X = T.tensor(np.random.random(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
    for i in range(T.ndim(X)):
        assert_array_equal(X, fold(unfold(X, i), i, X.shape))


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
    assert_array_equal(tensor_to_vec(X), true_res)


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
    assert_array_equal(X, vec_to_tensor(vec, X.shape))

    # Convert to vector and back to tensor
    X = T.tensor(np.random.random((3, 4, 5, 2)))
    vec = tensor_to_vec(X)
    reconstructed = vec_to_tensor(vec, X.shape)
    assert_array_equal(X, reconstructed)


def test_partial_unfold():
    """Test for partial_unfold

    Notes
    -----
    Assumes that the standard unfold is correct!
    """
    X = T.reshape(T.arange(24), (3, 4, 2))
    n_samples = 3
    ###################################
    # Samples are the first dimension #
    ###################################
    tensor = T.tensor(np.concatenate([np.arange(24).reshape((1, 3, 4, 2))+i\
                                      for i in range(n_samples)]))
    t = T.tensor(X)
    # We created here a tensor with 3 samples, each sample being similar to X
    for i in range(T.ndim(X)):  # test for each mode
        unfolded = partial_unfold(tensor, i, skip_begin=1)
        unfolded_X = unfold(t, i)
        for j in range(n_samples):  # test for each sample
            assert_array_equal(unfolded[j], unfolded_X+j)
    # Test for raveled tensor
    for i in range(T.ndim(X)):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_begin=1, ravel_tensors=True)
        unfolded_X = T.reshape(unfold(t, i), (-1, ))
        for j in range(n_samples):  # test for each sample
            assert_array_equal(unfolded[j], unfolded_X + j)

    ##################################
    # Samples are the last dimension #
    ##################################
    tensor = T.tensor(np.concatenate([np.arange(24).reshape((3, 4, 2, 1))+i\
                                      for i in range(n_samples)], axis=-1))
    for i in range(T.ndim(X)):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_end=1, skip_begin=0)
        unfolded_X = unfold(t, i)
        for j in range(n_samples):  # test for each sample
            assert_array_equal(T.transpose(T.transpose(unfolded)[j]), unfolded_X+j)

    # Test for raveled tensor
    for i in range(T.ndim(X)):  # test for each mode
        unfolded = partial_unfold(tensor, mode=i, skip_end=1, skip_begin=0, ravel_tensors=True)
        unfolded_X = T.reshape(unfold(t, i), (-1, ))
        for j in range(n_samples):  # test for each sample
            assert_array_equal(T.transpose(unfolded)[j], unfolded_X+j)

def test_partial_fold():
    """Test for partial_fold

    Assumes partial unfolding works and check that
    refolding partially folded tensors results in
    the original tensor.
    """
    X = T.reshape(T.arange(24), (3, 4, 2))
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
        assert_array_equal(folded[i], X)

    shape = [3, 4, 5, 6]
    X = T.tensor(np.random.random(shape))
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
    tensor = T.tensor(np.concatenate([X[None, ...]+i for i in range(n_samples)]))
    #we created here a tensor with 3 samples, each sample being similar to X
    vectorised = partial_tensor_to_vec(tensor, skip_begin=1)
    vec_X = tensor_to_vec(T.tensor(X))
    for j in range(n_samples): # test for each sample
        assert_array_equal(vectorised[j], vec_X+j)

    ##################################
    # Samples are the last dimension #
    ##################################
    tensor = T.tensor(np.concatenate([X[..., None]+i for i in range(n_samples)], axis=-1))
    vectorised = partial_tensor_to_vec(tensor, skip_end=1, skip_begin=0)
    vec_X = tensor_to_vec(T.tensor(X))
    for j in range(n_samples): # test for each sample
        assert_array_equal(T.transpose(vectorised)[j], vec_X+j)


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
        assert_array_equal(folded[i], X+i)

    shape = [3, 4, 5, 6]
    X = T.tensor(np.random.random(shape))
    for i in [0, 1]:
        vec = partial_tensor_to_vec(X, skip_begin=i, skip_end=(1-i))
        ten = partial_vec_to_tensor(vec, shape=shape, skip_begin=i, skip_end=(1-i))
        assert_array_equal(X, ten)


def test_svd():
    """Test for the SVD functions"""
    tol = 0.1
    tol_orthogonality = 0.01

    for name, svd_fun in T.SVD_FUNS.items():
        sizes = [(100, 100), (100, 5), (10, 10), (10, 4), (5, 100)]
        n_eigenvecs = [90, 4, 5, 4, 5]

        for s, n in zip(sizes, n_eigenvecs):
            matrix = np.random.random(s)
            matrix_backend = T.tensor(matrix)
            fU, fS, fV = svd_fun(matrix_backend, n_eigenvecs=n)
            U, S, V = svd(matrix)
            U, S, V = U[:, :n], S[:n], V[:n, :]

            assert_array_almost_equal(np.abs(S), T.abs(fS), decimal=3,
                err_msg='eigenvals not correct for "{}" svd fun VS svd and backend="{}, for {} eigenenvecs, and size {}".'.format(
                        name, tl.get_backend(), n, s))

            # True reconstruction error (based on numpy SVD)
            true_rec_error = np.sum((matrix - np.dot(U, S.reshape((-1, 1))*V))**2)
            # Reconstruction error with the backend's SVD
            rec_error = T.sum((matrix_backend - T.dot(fU, T.reshape(fS, (-1, 1))*fV))**2)
            # Check that the two are similar
            assert_(true_rec_error - rec_error <= tol,
                msg='Reconstruction not correct for "{}" svd fun VS svd and backend="{}, for {} eigenenvecs, and size {}".'.format(
                        name, tl.get_backend(), n, s))

            # Check for orthogonality when relevant
            if name != 'symeig_svd':
                left_orthogonality_error = T.norm(T.dot(T.transpose(fU), fU) - T.eye(n))
                assert_(left_orthogonality_error <= tol_orthogonality,
                    msg='Left eigenvecs not orthogonal for "{}" svd fun VS svd and backend="{}, for {} eigenenvecs, and size {}".'.format(
                            name, tl.get_backend(), n, s))
                right_orthogonality_error = T.norm(T.dot(T.transpose(fU), fU) - T.eye(n))
                assert_(right_orthogonality_error <= tol_orthogonality,
                    msg='Right eigenvecs not orthogonal for "{}" svd fun VS svd and backend="{}, for {} eigenenvecs, and size {}".'.format(
                        name, tl.get_backend(), n, s))

        # Should fail on non-matrices
        with assert_raises(ValueError):
            tensor = T.tensor(np.random.random((3, 3, 3)))
            svd_fun(tensor)


def test_shape():
    A = T.arange(3*4*5)

    shape1 = (3*4,5)
    A1 = T.reshape(A, shape1)
    assert_equal(T.shape(A1), shape1)

    shape2 = (3,4,5)
    A2 = T.reshape(A, shape2)
    assert_equal(T.shape(A2), shape2)


def test_ndim():
    A = T.arange(3*4*5)
    assert_equal(T.ndim(A), 1)

    shape1 = (3*4,5)
    A1 = T.reshape(A, shape1)
    assert_equal(T.ndim(A1), 2)

    shape2 = (3,4,5)
    A2 = T.reshape(A, shape2)
    assert_equal(T.ndim(A2), 3)


def test_norm():
    v = T.tensor([1., 2., 3.])
    assert_equal(T.norm(v,1), 6)

    A = T.reshape(T.arange(6), (3,2))
    assert_equal(T.norm(A, 1), 15)

    column_norms1 = T.norm(A, 1, axis=0)
    row_norms1 = T.norm(A, 1, axis=1)
    assert_array_equal(column_norms1, T.tensor([6., 9]))
    assert_array_equal(row_norms1, T.tensor([1, 5, 9]))

    column_norms2 = T.norm(A, 2, axis=0)
    row_norms2 = T.norm(A, 2, axis=1)
    assert_array_almost_equal(column_norms2, T.tensor([4.47213602, 5.91608]))
    assert_array_almost_equal(row_norms2, T.tensor([1., 3.60555124, 6.40312433]))

    # limit as order->oo is the oo-norm
    column_norms10 = T.norm(A, 10, axis=0)
    row_norms10 = T.norm(A, 10, axis=1)
    assert_array_almost_equal(column_norms10, T.tensor([4.00039053, 5.00301552]))
    assert_array_almost_equal(row_norms10, T.tensor([1., 3.00516224, 5.05125666]))

    column_norms_oo = T.norm(A, 'inf', axis=0)
    row_norms_oo = T.norm(A, 'inf', axis=1)
    assert_array_equal(column_norms_oo, T.tensor([4, 5]))
    assert_array_equal(row_norms_oo, T.tensor([1, 3, 5]))


def test_where():
    # 1D
    shape = (2*3*4,); N = np.prod(shape)
    X = T.arange(N)
    zeros = T.zeros(X.shape)
    ones = T.ones(X.shape)
    out = T.where(X < 2*3, zeros, ones)
    for i in range(N):
        if i < 2*3:
            assert_equal(out[i], 0, 'Unexpected result on vector for element {}'.format(i))
        else:
            assert_equal(out[i], 1, 'Unexpected result on vector for element {}'.format(i))

    # 2D
    shape = (2*3,4); N = np.prod(shape)
    X = T.reshape(T.arange(N), shape)
    zeros = T.zeros(X.shape)
    ones = T.ones(X.shape)
    out = T.where(X < 2*3, zeros, ones)
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = i*shape[1] + j
            if index < 2*3:
                assert_equal(out[i,j], 0, 'Unexpected result on matrix')
            else:
                assert_equal(out[i,j], 1, 'Unexpected result on matrix')

    # 3D
    shape = (2,3,4); N = np.prod(shape)
    X = T.reshape(T.arange(N), shape)
    zeros = T.zeros(X.shape)
    ones = T.ones(X.shape)
    out = T.where(X < 2*3, zeros, ones)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                index = (i*shape[1] + j)*shape[2] + k
                if index < 2*3:
                    assert_equal(out[i,j, k], 0, 'Unexpected result on matrix')
                else:
                    assert_equal(out[i,j, k], 1, 'Unexpected result on matrix')

    # random testing against Numpy's output
    shapes = (16,8,4,2)
    for order in range(1,5):
        shape = shapes[:order]
        tensor = T.tensor(np.random.randn(*shape))
        args = (tensor < 0, T.zeros(shape), T.ones(shape))
        result = T.where(*args)
        expected = np.where(*map(T.to_numpy, args))
        assert_array_equal(result, expected)


def test_qr():
    M = 8; N = 5
    A = T.tensor(np.random.random((M,N)))
    Q, R = T.qr(A)

    assert T.shape(Q) == (M,N), 'Unexpected shape'
    assert T.shape(R) == (N,N), 'Unexpected shape'

    # assert that the columns of Q are orthonormal
    Q_column_norms = T.norm(Q, 2, axis=0)
    assert_array_almost_equal(Q_column_norms, T.ones(N))
    for i in range(N):
        for j in range(i):
            dot_product = T.to_numpy(T.dot(Q[:,i], Q[:,j]))
            assert abs(dot_product) < 1e-6, 'Columns of Q not orthogonal'

    A_reconstructed = T.dot(Q, R)
    assert_array_almost_equal(A, A_reconstructed)


def test_prod():
    v = T.tensor([3, 4, 5])
    x = T.to_numpy(T.prod(v))
    assert_equal(x, 60)
