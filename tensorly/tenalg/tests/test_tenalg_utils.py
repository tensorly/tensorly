from ...testing import assert_
from ..tenalg_utils import _validate_contraction_modes

def test_validate_contraction_modes():
    """Test _validate_contraction_modes"""
    shape1 = (3, 4, 5, 2, 2, 4)
    shape2 = (3, 3, 5, 2, 3, 4)
    
    true_modes1 = [0, 1, 4]
    true_modes2 = [1, 5, 3]
    modes = ((0, 1, 4), (1, 5, 3))
    modes1, modes2 = _validate_contraction_modes(shape1, shape2, modes, batched_modes=False)
    assert_((true_modes1, true_modes2) == (modes1, modes2), 
            f'Got ({modes1}, {modes2}), but expected ({true_modes1}, {true_modes2}).')
    
    true_batched_modes1 = [0, 2]
    true_batched_modes2 = [0, 2]
    modes = ((0, 2), (0, 2))
    batched_modes1, batched_modes2 = _validate_contraction_modes(shape1, shape2, modes, batched_modes=True)
    assert_((true_batched_modes1, true_batched_modes2) == (batched_modes1, batched_modes2), 
            f'Got ({batched_modes1}, {batched_modes2}), but expected ({true_batched_modes1}, {true_batched_modes2}).')
    
    true_batched_modes1 = [0, 2, 5]
    true_batched_modes2 = [0, 2, 5]
    modes = (0, 2, 5)
    batched_modes1, batched_modes2 = _validate_contraction_modes(shape1, shape2, modes, batched_modes=True)
    assert_((true_batched_modes1, true_batched_modes2) == (batched_modes1, batched_modes2), 
            f'Got ({batched_modes1}, {batched_modes2}), but expected ({true_batched_modes1}, {true_batched_modes2}).')

    true_batched_modes1 = [0]
    true_batched_modes2 = [0]
    modes = 0
    batched_modes1, batched_modes2 = _validate_contraction_modes(shape1, shape2, modes, batched_modes=True)
    assert_((true_batched_modes1, true_batched_modes2) == (batched_modes1, batched_modes2), 
            f'Got ({batched_modes1}, {batched_modes2}), but expected ({true_batched_modes1}, {true_batched_modes2}).')

    # Check that negative dims are made positive
    true_modes1 = [0, 4, 5]
    true_modes2 = [1, 3, 5]
    modes = ((0, -2, -1), (1, -3, -1))
    modes1, modes2 = _validate_contraction_modes(shape1, shape2, modes, batched_modes=False)
    assert_((true_modes1, true_modes2) == (modes1, modes2), 
            f'Got ({modes1}, {modes2}), but expected ({true_modes1}, {true_modes2}).')

test_validate_contraction_modes()