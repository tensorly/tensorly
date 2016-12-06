from ..synthetic import gen_image, low_rank_matrix 
from numpy.testing import assert_array_equal, assert_equal 
from numpy.linalg import matrix_rank 
import numpy as np 


def test_gen_image():
    """Test for image_weight"""
    # Swiss
    true_res = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                         [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                         [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                         [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    res = gen_image(region='swiss', image_height=10, image_width=10, weight_value=1)
    assert_array_equal(true_res, res)

    # Rectangle
    true_res = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    res = gen_image(region='rectangle', image_height=10, image_width=10, weight_value=1)
    assert_array_equal(true_res, res)



 
def test_low_rank_matrix(): 
    """Test for low_rank_matrix""" 
    rank = 5 
    mat = low_rank_matrix((100, 1000), rank) 
    assert_equal(matrix_rank(mat), rank) 

