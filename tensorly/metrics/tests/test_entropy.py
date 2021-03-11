import tensorly as tl
from ..entropy import vonneumann_entropy, mps_entanglement_entropy
from ..entropy import tt_vonneumann_entropy, cp_vonneumann_entropy
from ...decomposition import parafac, matrix_product_state
from tensorly.testing import assert_array_almost_equal

def test_vonneumann_entropy_pure_state():
    """Test for vonneumann_entropy on 2-dimensional tensors.
    This test checks that pure states have a VNE of zero.
    """
    state = tl.randn((8, 1))
    state = state/tl.norm(state)
    mat_pure = tl.dot(state, tl.transpose(state))
    tl_vne = vonneumann_entropy(mat_pure)
    assert_array_almost_equal(tl_vne, 0, decimal=3)

def test_tt_vonneumann_entropy_pure_state():
    """Test for tt_vonneumann_entropy TT tensors.
    This test checks that pure states have a VNE of zero.
    """
    state = tl.randn((8, 1))
    state = state/tl.norm(state)
    mat_pure = tl.reshape(tl.dot(state, tl.transpose(state)), (2, 2, 2, 2, 2, 2))
    mat_pure = matrix_product_state(mat_pure, rank=[1, 2, 2, 2, 2, 2, 1])
    tl_vne = tt_vonneumann_entropy(mat_pure)
    assert_array_almost_equal(tl_vne, 0, decimal=3)

def test_cp_vonneumann_entropy_pure_state():
    """Test for cp_vonneumann_entropy on 2-dimensional CP tensors.
    This test checks that pure states have a VNE of zero.
    """
    state = tl.randn((8, 1))
    state = state/tl.norm(state)
    mat_pure = tl.dot(state, tl.transpose(state))
    mat = parafac(mat_pure, rank=1, normalize_factors=True)
    tl_vne = cp_vonneumann_entropy(mat)
    assert_array_almost_equal(tl_vne, 0, decimal=3)

def test_vonneumann_entropy_mixed_state():
    """Test for vonneumann_entropy on 2-dimensional tensors.
    This test checks that the VNE of mixed states is calculated correctly.
    """
    state1 = tl.tensor([[0.03004805, 0.42426117, 0.5483771 , 0.4784077 , 0.25792725, 0.34388784, 0.99927586, 0.96605812]])
    state1 = state1/tl.norm(state1)
    state2 = tl.tensor([[0.84250089, 0.43429687, 0.26551928, 0.18262211, 0.55584835, 0.2565509 , 0.33197401, 0.97741178]])
    state2 = state2/tl.norm(state2)
    mat_mixed = tl.tensor((tl.dot(tl.transpose(state1), state1) + tl.dot(tl.transpose(state2), state2))/2.)
    actual_vne = 0.5546
    tl_vne = vonneumann_entropy(mat_mixed)
    assert_array_almost_equal(tl_vne, actual_vne, decimal=3)

def test_tt_vonneumann_entropy_mixed_state():
    """Test for tt_vonneumann_entropy on TT tensors.
    This test checks that the VNE of mixed states is calculated correctly.
    """
    state1 = tl.tensor([[0.03004805, 0.42426117, 0.5483771 , 0.4784077 , 0.25792725, 0.34388784, 0.99927586, 0.96605812]])
    state1 = state1/tl.norm(state1)
    state2 = tl.tensor([[0.84250089, 0.43429687, 0.26551928, 0.18262211, 0.55584835, 0.2565509 , 0.33197401, 0.97741178]])
    state2 = state2/tl.norm(state2)
    mat_mixed = tl.tensor((tl.dot(tl.transpose(state1), state1) + tl.dot(tl.transpose(state2), state2))/2.)
    actual_vne = 0.5546
    tt_mixed = tl.reshape(mat_mixed, (2, 2, 2, 2, 2, 2))
    tt_mixed = matrix_product_state(tt_mixed, rank=[1, 2, 4, 8, 4, 2, 1])
    tl_vne = tt_vonneumann_entropy(tt_mixed)
    assert_array_almost_equal(tl_vne, actual_vne, decimal=3)

def test_cp_vonneumann_entropy_mixed_state():
    """Test for cp_vonneumann_entropy on CP tensors. 
    This test checks that the VNE of mixed states is calculated correctly.
    """
    state1 = tl.tensor([[0.03004805, 0.42426117, 0.5483771 , 0.4784077 , 0.25792725, 0.34388784, 0.99927586, 0.96605812]])
    state1 = state1/tl.norm(state1)
    state2 = tl.tensor([[0.84250089, 0.43429687, 0.26551928, 0.18262211, 0.55584835, 0.2565509 , 0.33197401, 0.97741178]])
    state2 = state2/tl.norm(state2)
    mat_mixed = tl.tensor((tl.dot(tl.transpose(state1), state1) + tl.dot(tl.transpose(state2), state2))/2.)
    actual_vne = 0.5546
    mat = parafac(tl.tensor(mat_mixed), rank=2, normalize_factors=True)
    mat_unnorm = parafac(tl.tensor(mat_mixed), rank=2, normalize_factors=False)
    tl_vne = cp_vonneumann_entropy(mat)
    tl_vne_unnorm = cp_vonneumann_entropy(mat_unnorm)
    assert_array_almost_equal(tl_vne, actual_vne, decimal=3)
    assert_array_almost_equal(tl_vne_unnorm, actual_vne, decimal=3)

def test_mps_entanglement_entropy():
    """Test for the tt_mps_entanglement_entropy on TT tensors.
    This test checks that the EE of both product and entangled states is calculated correctly.
    """
    #Product state
    mps = tl.tensor([1,0,0,0,0,0,0,0], dtype=tl.float32)
    mps = tl.reshape(mps, (2,2,2))
    mps = matrix_product_state(mps, rank=[1,2,2,1])
    tl_ee = mps_entanglement_entropy(mps, 1)
    assert_array_almost_equal(tl_ee, 0, decimal=3)

    #Entangled state
    mps = tl.tensor([1,0,0,0,0,0,0,1], dtype=tl.float32)
    mps = mps/tl.norm(mps)
    mps = tl.reshape(mps, (2,2,2))
    mps = matrix_product_state(mps, rank=[1,2,2,1])
    tl_ee = mps_entanglement_entropy(mps, 1)
    assert_array_almost_equal(tl_ee, 1, decimal=3)
