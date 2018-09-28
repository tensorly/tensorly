import numpy as np

import tensorly as tl
import tensorly.backend as T
from tensorly.tenalg import khatri_rao
from tensorly.kruskal_tensor import (kruskal_to_tensor, kruskal_to_unfolded,
                                     kruskal_to_vec)
from tensorly.base import unfold, tensor_to_vec
from tensorly.testing import assert_array_equal, assert_array_almost_equal


# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


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
    res = kruskal_to_tensor(U)
    assert_array_equal(res, true_res, err_msg='Khatri-rao incorrectly transformed into full tensor.')

    columns = 4
    rows = [3, 4, 2]
    matrices = [tl.tensor(np.arange(k * columns).reshape((k, columns))) for k in rows]
    tensor = kruskal_to_tensor(matrices)
    for i in range(len(rows)):
        unfolded = unfold(tensor, mode=i)
        U_i = matrices.pop(i)
        reconstructed = tl.dot(U_i, tl.transpose(khatri_rao(matrices)))
        assert_array_almost_equal(reconstructed, unfolded)
        matrices.insert(i, U_i)

def test_kruskal_to_tensor_with_weights():
    A = tl.reshape(tl.arange(1,5), (2,2))
    B = tl.reshape(tl.arange(5,9), (2,2))
    weights = tl.tensor([2,-1])

    out = kruskal_to_tensor([A,B], weights=weights)
    expected = tl.tensor([[-2,-2], [6, 10]])  # computed by hand
    assert_array_equal(out, expected)


def test_kruskal_to_unfolded():
    """Test for kruskal_to_unfolded.
        !!Assumes that kruskal_to_tensor and unfold are properly tested and work!!
    """
    U1 = np.reshape(np.arange(1, 10), (3, 3))
    U2 = np.reshape(np.arange(10, 22), (4, 3))
    U3 = np.reshape(np.arange(22, 28), (2, 3))
    U4 = np.reshape(np.arange(28, 34), (2, 3))
    U = [tl.tensor(t) for t in [U1, U2, U3, U4]]
    full_tensor = kruskal_to_tensor(U)
    for mode in range(4):
        true_res = unfold(full_tensor, mode)
        res = kruskal_to_unfolded(U, mode)
        assert_array_equal(true_res, res, err_msg='khatri_rao product unfolded incorrectly for mode {}.'.format(mode))


def test_kruskal_to_vec():
    """Test for kruskal_to_vec"""
    U1 = np.reshape(np.arange(1, 10), (3, 3))
    U2 = np.reshape(np.arange(10, 22), (4, 3))
    U3 = np.reshape(np.arange(22, 28), (2, 3))
    U4 = np.reshape(np.arange(28, 34), (2, 3))
    U = [tl.tensor(t) for t in [U1, U2, U3, U4]]
    full_tensor = kruskal_to_tensor(U)
    true_res = tensor_to_vec(full_tensor)
    res = kruskal_to_vec(U)
    assert_array_equal(true_res, res, err_msg='khatri_rao product converted incorrectly to vec.')
