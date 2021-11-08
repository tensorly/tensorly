import pytest
import tensorly as tl
from tensorly import random
from ..tt_matrix import tt_matrix_to_matrix, tt_matrix_to_tensor, tt_matrix_to_vec

skip_mxnet= pytest.mark.skipif(tl.get_backend() == "mxnet", 
                 reason="MXNet currently does not support transpose for tensors of order > 6.")

 # TODO: Remove once MXNet supports transpose for > 6th order tensors
@skip_mxnet
def test_tt_matrix_manipulation():
    """Test for tt_matrix manipulation"""
    shape = (2, 2, 2, 3, 3, 3)
    n_rows, n_cols = 8, 27
    tt_matrix = random.random_tt_matrix(shape, rank=2, full=False)
    rec = tt_matrix_to_tensor(tt_matrix)
    assert(tl.shape(rec) == shape)

    mat = tt_matrix_to_matrix(tt_matrix)
    assert(tl.shape(mat) == (n_rows, n_cols))

    vec = tt_matrix_to_vec(tt_matrix)
    assert(tl.shape(vec) == (n_rows*n_cols,))