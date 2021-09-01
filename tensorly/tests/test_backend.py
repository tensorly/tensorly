from concurrent.futures import ThreadPoolExecutor

import pytest
from time import time
import numpy as np
from scipy.linalg import svd

import tensorly as tl
from .. import backend as T
from ..testing import (assert_array_equal, assert_equal, assert_,
                              assert_array_almost_equal, assert_raises)

# Author: Jean Kossaifi


def test_set_backend():
    torch = pytest.importorskip('torch')

    toplevel_backend = tl.get_backend()

    # Set in context manager
    with tl.backend_context('numpy'):
        assert tl.get_backend() == 'numpy'
        assert isinstance(tl.tensor([1, 2, 3]), np.ndarray)
        assert isinstance(T.tensor([1, 2, 3]), np.ndarray)
        assert tl.float32 is T.float32 is np.float32

        with tl.backend_context('pytorch'):
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
    tl.set_backend('pytorch')
    assert tl.get_backend() == 'pytorch'
    tl.set_backend(toplevel_backend)

    assert tl.get_backend() == toplevel_backend

    # Improper name doesn't reset backend
    with assert_raises(ValueError):
        tl.set_backend('not-a-real-backend')
    assert tl.get_backend() == toplevel_backend


def test_set_backend_local_threadsafe():
    pytest.importorskip('torch')

    global_default = tl.get_backend()

    with ThreadPoolExecutor(max_workers=1) as executor:

        with tl.backend_context('numpy', local_threadsafe=True):
            assert tl.get_backend() == 'numpy'
            # Changes only happen locally in this thread
            assert executor.submit(tl.get_backend).result() == global_default

        # Set the global default backend
        try:
            tl.set_backend('pytorch', local_threadsafe=False)

            # Changed toplevel default in all threads
            assert executor.submit(tl.get_backend).result() == 'pytorch'

            with tl.backend_context('numpy', local_threadsafe=True):
                assert tl.get_backend() == 'numpy'

                def check():
                    assert tl.get_backend() == 'pytorch'
                    with tl.backend_context('numpy', local_threadsafe=True):
                        assert tl.get_backend() == 'numpy'
                    assert tl.get_backend() == 'pytorch'

                executor.submit(check).result()
        finally:
            tl.set_backend(global_default, local_threadsafe=False)
            executor.submit(tl.set_backend,global_default).result()

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


def test_svd_time():
    """Test SVD time
    
    SVD shouldn't be slow for tall and skinny matrices
    if n_eigenvec == min(matrix.shape)
    """
    M = tl.tensor(np.random.random_sample((4, 10000)))
    t = time()
    _ = tl.partial_svd(M, 4)
    t = time() - t
    assert_(t <= 0.1, f'Partial_SVD took too long, maybe full_matrices set wrongly')

    M = tl.tensor(np.random.random_sample((10000, 4)))
    t = time()
    _ = tl.partial_svd(M, 4)
    t = time() - t
    assert_(t <= 0.1, f'Partial_SVD took too long, maybe full_matrices set wrongly')


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
            left_orthogonality_error = T.norm(T.dot(T.transpose(fU), fU) - T.eye(n))
            assert_(left_orthogonality_error <= tol_orthogonality,
                msg='Left eigenvecs not orthogonal for "{}" svd fun VS svd and backend="{}, for {} eigenenvecs, and size {}".'.format(
                        name, tl.get_backend(), n, s))
            right_orthogonality_error = T.norm(T.dot(fV, T.transpose(fV)) - T.eye(n))
            assert_(right_orthogonality_error <= tol_orthogonality,
                msg='Right eigenvecs not orthogonal for "{}" svd fun VS svd and backend="{}, for {} eigenenvecs, and size {}".'.format(
                    name, tl.get_backend(), n, s))

        # Should fail on non-matrices
        with assert_raises(ValueError):
            tensor = T.tensor(np.random.random((3, 3, 3)))
            svd_fun(tensor)

        # Test for singular matrices (some eigenvals will be zero)
        # Rank at most 5
        matrix = tl.dot(tl.randn((20, 5), seed=12), tl.randn((5, 20), seed=23))
        U, S, V = tl.partial_svd(matrix, n_eigenvecs=6, random_state=0)
        true_rec_error = tl.sum((matrix - tl.dot(U, tl.reshape(S, (-1, 1))*V))**2)
        assert_(true_rec_error <= tol)
        assert_(np.isfinite(T.to_numpy(U)).all(), msg="Left singular vectors are not finite")
        assert_(np.isfinite(T.to_numpy(V)).all(), msg="Right singular vectors are not finite")

        # Test orthonormality when  max_dim > n_eigenvecs > matrix_rank
        matrix = tl.dot(tl.randn((4, 2), seed=1), tl.randn((2, 4), seed=12))
        U, S, V = tl.partial_svd(matrix, n_eigenvecs=3, random_state=0)
        left_orthogonality_error = T.norm(T.dot(T.transpose(U), U) - T.eye(3))
        assert_(left_orthogonality_error <= tol_orthogonality)
        right_orthogonality_error = T.norm(T.dot(V, T.transpose(V)) - T.eye(3))
        assert_(right_orthogonality_error <= tol_orthogonality)

        # Test if partial_svd returns the same result for the same setting
        matrix = T.tensor(np.random.random((20, 5)))
        random_state = np.random.RandomState(0)
        U1, S1, V1 = tl.partial_svd(matrix, n_eigenvecs=2, random_state=random_state)
        U2, S2, V2 = tl.partial_svd(matrix, n_eigenvecs=2, random_state=0)
        assert_array_equal(U1, U2)
        assert_array_equal(S1, S2)
        assert_array_equal(V1, V2)


def test_randomized_range_finder():
    size = (7, 5)
    A = T.randn(size)
    Q = T.randomized_range_finder(A, n_dims=min(size))
    assert_array_almost_equal(A, tl.dot(tl.dot(Q, tl.transpose(T.conj(Q))), A))


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


def test_clip():
    """Test that clip can work with single arguments"""
    X = T.tensor([0.0, -1.0, 1.0])
    X_low = T.tensor([0.0, 0.0, 1.0])
    X_high = T.tensor([0.0, -1.0, 0.0])
    assert_array_equal(tl.clip(X, a_min=0.0), X_low)
    assert_array_equal(tl.clip(X, a_max=0.0), X_high)


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


def test_matmul():
    # 1-dim x n-dim
    a = tl.randn((4,))
    b = tl.randn((2, 3, 4, 5))
    res = tl.matmul(a, b)
    assert_equal(res.shape, (2, 3, 5))

    # n_dim x 1-dim
    a = tl.randn((5,))
    res = tl.matmul(b, a)
    assert_equal(res.shape, (2, 3, 4))

    # n-dim x n-dim
    a = tl.randn((2, 3, 6, 4))
    res = tl.matmul(a, b)
    assert_equal(res.shape, (2, 3, 6, 5))


def test_lstsq():
    m, n, k = 4, 3, 2

    # test dimensions
    a = T.randn((m, n))
    b = T.randn((m, k))
    x, res = T.lstsq(a, b)
    assert_equal(x.shape, (n, k))

    # test residuals
    assert_array_almost_equal(T.norm(T.dot(a, x) - b, axis=0) ** 2, res)
    rank = 2
    a = T.dot(T.randn((m, rank)), T.randn((rank, n)))
    _, res = T.lstsq(a, b)
    assert_array_almost_equal(tl.tensor([]), res)

    # test least squares solution
    a = T.randn((m, n))
    x = T.randn((n, ))
    b = T.dot(a, x)
    x_lstsq, res = T.lstsq(a, b)
    assert_array_almost_equal(T.dot(a, x_lstsq), b, decimal=5)


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


def test_index_update():
    np_tensor = np.random.random((3, 5)).astype(dtype=np.float32)
    tensor = tl.tensor(np.copy(np_tensor))
    np_insert = np.random.random((3, 2)).astype(dtype=np.float32)
    insert = tl.tensor(np.copy(np_insert))

    np_tensor[:, 1:3] = np_insert
    tensor = tl.index_update(tensor, tl.index[:, 1:3], insert)
    assert_array_equal(np_tensor, tensor)

    np_tensor = np.random.random((3, 5)).astype(dtype=np.float32)
    tensor = tl.tensor(np.copy(np_tensor))
    np_tensor[2, :] = 2
    tensor = tl.index_update(tensor, tl.index[2, :], 2)
    assert_array_equal(np_tensor, tensor)

