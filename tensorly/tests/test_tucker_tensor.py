import numpy as np
from tensorly import backend as T
from tensorly.base import unfold, tensor_to_vec
from tensorly.tucker_tensor import tucker_to_tensor, tucker_to_unfolded, tucker_to_vec
from tensorly.tenalg import kronecker
from tensorly.testing import assert_array_equal, assert_array_almost_equal


def test_tucker_to_tensor():
    """Test for tucker_to_tensor"""
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
    ranks = [2, 3, 4]
    U = [T.tensor(np.arange(R * s).reshape((R, s))) for (R, s) in zip(ranks, T.shape(X))]
    true_res = np.array([[[390, 1518, 2646, 3774],
                         [1310, 4966, 8622, 12278],
                         [2230, 8414, 14598, 20782]],
                        [[1524, 5892, 10260, 14628],
                         [5108, 19204, 33300, 47396],
                         [8692, 32516, 56340, 80164]]])
    res = tucker_to_tensor(X, U)
    assert_array_equal(true_res, res)


def test_tucker_to_unfolded():
    """Test for tucker_to_unfolded

    Notes
    -----
    Assumes that tucker_to_tensor is properly tested
    """
    G = T.tensor(np.random.random((4, 3, 5, 2)))
    ranks = [2, 2, 3, 4]
    U = [T.tensor(np.random.random((ranks[i], G.shape[i]))) for i in range(T.ndim(G))]
    full_tensor = tucker_to_tensor(G, U)
    for mode in range(T.ndim(G)):
        assert_array_almost_equal(tucker_to_unfolded(G, U, mode), unfold(full_tensor, mode))
        assert_array_almost_equal(tucker_to_unfolded(G, U, mode),
                                    T.dot(T.dot(U[mode], unfold(G, mode)), T.transpose(kronecker(U, skip_matrix=mode))),
                                    decimal=5)


def test_tucker_to_vec():
    """Test for tucker_to_vec

    Notes
    -----
    Assumes that tucker_to_tensor works correctly
    """
    G = T.tensor(np.random.random((4, 3, 5, 2)))
    ranks = [2, 2, 3, 4]
    U = [T.tensor(np.random.random((ranks[i], G.shape[i]))) for i in range(T.ndim(G))]
    vec = tensor_to_vec(tucker_to_tensor(G, U))
    assert_array_almost_equal(tucker_to_vec(G, U), vec)
    assert_array_almost_equal(tucker_to_vec(G, U), T.dot(kronecker(U), tensor_to_vec(G)), decimal=5)
