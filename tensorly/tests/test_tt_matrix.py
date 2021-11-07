import tensorly as tl
from tensorly import random
from ..tt_matrix import tt_matrix_to_matrix, tt_matrix_to_tensor, tt_matrix_to_vec

def test_tt_matrix_manipulation():
    """Test for tt_matrix manipulation"""
    shape = (2, 2, 3, 3) # Revert to (2, 2, 2, 3, 3, 3) once MXNet supports transpose for > 6th order tensors
    tt_matrix = random.random_tt_matrix(shape, rank=2, full=False)
    rec = tt_matrix_to_tensor(tt_matrix)
    assert(tl.shape(rec) == shape)

    mat = tt_matrix_to_matrix(tt_matrix)
    assert(tl.shape(mat) == (4, 9))

    vec = tt_matrix_to_vec(tt_matrix)
    assert(tl.shape(vec) == (4*9,))