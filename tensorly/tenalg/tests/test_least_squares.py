from ... import backend as T
from ..least_squares import solve_least_squares
from ...testing import assert_array_equal


def test_solve_least_squares():
    A = T.tensor([[1, 0], [0, 1]])
    B = T.tensor([[1, 2], [3, 4]])
    res = solve_least_squares(A, B)
    true_res = B
    assert_array_equal(true_res, res)
