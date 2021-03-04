import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import tensorly as tl
from ..entropy import vonNeumann_entropy
from ...decomposition import parafac

state = npr.rand(8, 1).astype(np.float32)
state = state/npl.norm(state)
mat_pure = tl.tensor(np.matmul(state, np.transpose(state)))

state = np.array([[0.03004805, 0.42426117, 0.5483771 , 0.4784077 , 0.25792725, 0.34388784, 0.99927586, 0.96605812]])
state = state/npl.norm(state)
state2 = np.array([[0.84250089, 0.43429687, 0.26551928, 0.18262211, 0.55584835, 0.2565509 , 0.33197401, 0.97741178]])
state2 = state2/npl.norm(state2)
mat_mixed = tl.tensor((np.matmul(state.T, state) + np.matmul(state2.T, state2))/2.)

def test_vonNeumann_entropy_pure_state():
    tl_vne = np.array(vonNeumann_entropy(mat_pure))
    assert np.allclose(tl_vne, np.array([0]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_pure_state_CP():
    mat = parafac(tl.tensor(mat_pure), rank=1, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(mat))
    assert np.allclose(tl_vne, np.array([0]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_mixed_state():
    tl_vne = np.array(vonNeumann_entropy(mat_mixed))
    print(tl_vne)
    assert np.allclose(tl_vne, np.array([0.5545]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_mixed_state_CP():
    mat = parafac(tl.tensor(mat_mixed), rank=2, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(mat))
    assert np.allclose(tl_vne, np.array([0.5545]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_mixed_state_CP_unnormalized_factors():
    mat = parafac(tl.tensor(mat_mixed), rank=2, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(mat))
    assert np.allclose(tl_vne, np.array([0.5545]), rtol=1e-03, atol=1e-03)
