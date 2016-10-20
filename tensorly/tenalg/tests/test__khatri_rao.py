from numpy.testing import assert_raises, assert_array_equal, assert_array_almost_equal
import numpy as np

from .._khatri_rao import khatri_rao


# Author: Jean Kossaifi


def test_khatri_rao():
    """Test for khatri_rao
    """
    columns = 4
    rows = [3, 4, 2]
    matrices = [np.arange(k * columns).reshape((k, columns)) for k in rows]
    res = khatri_rao(matrices)
    # resulting matrix must be of shape (prod(n_rows), n_columns)
    n_rows = 3 * 4 * 2
    n_columns = 4
    assert (res.shape[0] == n_rows)
    assert (res.shape[1] == n_columns)

    # fail case: all matrices must have same number of columns
    shapes = [[3, 4], [3, 4], [3, 2]]
    matrices = [np.arange(i * j).reshape((i, j)) for (i, j) in shapes]
    with assert_raises(ValueError):
        khatri_rao(matrices)

    # all matrices should be of dim 2...
    matrices = [np.eye(3), np.arange(3 * 2 * 2).reshape((3, 2, 2))]
    with assert_raises(ValueError):
        khatri_rao(matrices)

    # Classic example/test
    t1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    t2 = np.array([[1, 4, 7],
                   [2, 5, 8],
                   [3, 6, 9]])
    true_res = np.array([[1., 8., 21.],
                         [2., 10., 24.],
                         [3., 12., 27.],
                         [4., 20., 42.],
                         [8., 25., 48.],
                         [12., 30., 54.],
                         [7., 32., 63.],
                         [14., 40., 72.],
                         [21., 48., 81.]])
    reversed_true_res = np.array([[1., 8., 21.],
                                  [4., 20., 42.],
                                  [7., 32., 63.],
                                  [2., 10., 24.],
                                  [8., 25., 48.],
                                  [14., 40., 72.],
                                  [3., 12., 27.],
                                  [12., 30., 54.],
                                  [21., 48., 81.]])
    res = khatri_rao([t1, t2])
    assert_array_equal(res, true_res)
    reversed_res = khatri_rao([t1, t2], reverse=True)
    assert_array_equal(reversed_res, reversed_true_res)

    # A = np.hstack((np.eye(3), np.arange(3)[:, None]))
    A = np.array([[ 1.,  0.,  0.,  0.],
                  [ 0.,  1.,  0.,  1.],
                  [ 0.,  0.,  1.,  2.]])
    B = np.array([[ 1.,  0.,  0.,  3.],
                  [ 0.,  1.,  0.,  4.],
                  [ 0.,  0.,  1.,  5.]])
    true_res = np.array([[  1.,   0.,   0.,   0.],
                         [  0.,   0.,   0.,   0.],
                         [  0.,   0.,   0.,   0.],
                         [  0.,   0.,   0.,   3.],
                         [  0.,   1.,   0.,   4.],
                         [  0.,   0.,   0.,   5.],
                         [  0.,   0.,   0.,   6.],
                         [  0.,   0.,   0.,   8.],
                         [  0.,   0.,   1.,  10.]])
    assert_array_equal(khatri_rao([A, B]), true_res)

    U1 = np.reshape(np.arange(1, 10), (3, 3))
    U2 = np.reshape(np.arange(10, 22), (4, 3))
    U3 = np.reshape(np.arange(22, 28), (2, 3))
    U4 = np.reshape(np.arange(28, 34), (2, 3))
    U = [U1, U2, U3, U4]
    true_res = true_res = np.array([[  6160,        14674,        25920],
                                  [  6820,        16192,        28512],
                                  [  7000,        16588,        29160],
                                  [  7750,        18304,        32076],
                                  [  8008,        18676,        32400],
                                  [  8866,        20608,        35640],
                                  [  9100,        21112,        36450],
                                  [ 10075,        23296,        40095],
                                  [  9856,        22678,        38880],
                                  [ 10912,        25024,        42768],
                                  [ 11200,        25636,        43740],
                                  [ 12400,        28288,        48114],
                                  [ 11704,        26680,        45360],
                                  [ 12958,        29440,        49896],
                                  [ 13300,        30160,        51030],
                                  [ 14725,        33280,        56133],
                                  [ 24640,        36685,        51840],
                                  [ 27280,        40480,        57024],
                                  [ 28000,        41470,        58320],
                                  [ 31000,        45760,        64152],
                                  [ 32032,        46690,        64800],
                                  [ 35464,        51520,        71280],
                                  [ 36400,        52780,        72900],
                                  [ 40300,        58240,        80190],
                                  [ 39424,        56695,        77760],
                                  [ 43648,        62560,        85536],
                                  [ 44800,        64090,        87480],
                                  [ 49600,        70720,        96228],
                                  [ 46816,        66700,        90720],
                                  [ 51832,        73600,        99792],
                                  [ 53200,        75400,       102060],
                                  [ 58900,        83200,       112266],
                                  [ 43120,        58696,        77760],
                                  [ 47740,        64768,        85536],
                                  [ 49000,        66352,        87480],
                                  [ 54250,        73216,        96228],
                                  [ 56056,        74704,        97200],
                                  [ 62062,        82432,       106920],
                                  [ 63700,        84448,       109350],
                                  [ 70525,        93184,       120285],
                                  [ 68992,        90712,       116640],
                                  [ 76384,       100096,       128304],
                                  [ 78400,       102544,       131220],
                                  [ 86800,       113152,       144342],
                                  [ 81928,       106720,       136080],
                                  [ 90706,       117760,       149688],
                                  [ 93100,       120640,       153090],
                                  [103075,       133120,       168399]])
    res = khatri_rao(U)
    assert_array_equal(res, true_res)

    res_1 = khatri_rao(U, skip_matrix=1)
    res_2 = khatri_rao([U[0]] + U[2:])
    assert_array_equal(res_1, res_2)

    res_1 = khatri_rao(U, skip_matrix=0)
    res_2 = khatri_rao(U[1:])
    assert_array_equal(res_1, res_2)