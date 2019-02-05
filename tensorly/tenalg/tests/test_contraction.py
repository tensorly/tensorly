import numpy as np

import tensorly as tl
from ..contraction import contract
from ...testing import assert_array_almost_equal, assert_raises

def test_contract():
    shape = [3, 4, 2]
    vecs = [tl.tensor(np.random.random_sample((s))) for s in shape]
    tensor = tl.tensor(np.random.random_sample(shape))
    
    # Equivalence with inner product when contracting with self along all modes
    res = contract(tensor, [0, 1, 2], tensor, modes2=[0, 1, 2])
    true_res = tl.tenalg.inner(tensor, tensor, n_modes=3)
    assert_array_almost_equal(true_res, res)
        
    # Equivalence with n-mode-dot
    for mode, vec in enumerate(vecs):
        res = contract(tensor, mode, vec, 0)
        true_res = tl.tenalg.mode_dot(tensor, vec, mode)
        assert_array_almost_equal(true_res, res)
        
    # Multi-mode-dot
    res = contract(contract(tensor, 0, vecs[0], 0), 0, vecs[1], 0)
    true_res = tl.tenalg.multi_mode_dot(tensor, vecs[:2], [0, 1])

    # Wrong number of modes
    with assert_raises(ValueError):
        contract(tensor, [0, 2], tensor, modes2=[0, 1, 2])
    
    # size mismatch
    with assert_raises(ValueError):
        contract(tensor, 0, vecs[1], modes2=0)

