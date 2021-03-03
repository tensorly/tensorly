import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import tensorly as tl
import qutip
from ..entropy import vonNeumann_entropy
from ...decomposition import parafac

def test_vonNeumann_entropy_pure_state():
    state = npr.rand(16, 1)
    state = state/npl.norm(state)
    mat = np.matmul(state, np.transpose(state))
    tl_vne = np.array(vonNeumann_entropy(mat))
    dm_qutip = qutip.Qobj(mat)
    qutip_vne = qutip.entropy_vn(dm_qutip, base=2)
    assert np.allclose(tl_vne, qutip_vne)

def test_vonNeumann_entropy_pure_state_CP():
    state = npr.rand(16, 1)
    state = state/npl.norm(state)
    mat = np.matmul(state, np.transpose(state))
    tl_mat = parafac(tl.tensor(mat), rank=1, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(tl_mat))
    dm_qutip = qutip.Qobj(mat)
    qutip_vne = qutip.entropy_vn(dm_qutip, base=2)
    assert np.allclose(tl_vne, qutip_vne)

def test_vonNeumann_entropy_mixed_state():
    state1 = npr.rand(16, 1)
    state1 = state1/npl.norm(state1)
    mat1 = np.matmul(state1, np.transpose(state1))
    state2 = npr.rand(16, 1)
    state2 = state2/npl.norm(state2)
    mat2 = np.matmul(state2, np.transpose(state2))
    mat = (mat1 + mat2)/2.
    tl_vne = np.array(vonNeumann_entropy(mat))
    dm_qutip = qutip.Qobj(mat)
    qutip_vne = qutip.entropy_vn(dm_qutip, base=2)
    assert np.allclose(tl_vne, qutip_vne)

def test_vonNeumann_entropy_mixed_state_CP():
    state1 = npr.rand(16, 1)
    state1 = state1/npl.norm(state1)
    mat1 = np.matmul(state1, np.transpose(state1))
    state2 = npr.rand(16, 1)
    state2 = state2/npl.norm(state2)
    mat2 = np.matmul(state2, np.transpose(state2))
    mat = (mat1 + mat2)/2.
    tl_mat = parafac(tl.tensor(mat), rank=2, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(tl_mat))
    dm_qutip = qutip.Qobj(mat)
    qutip_vne = qutip.entropy_vn(dm_qutip, base=2)
    assert np.allclose(tl_vne, qutip_vne)

def test_vonNeumann_entropy_mixed_state_CP_unnormalized_factors():
    state1 = npr.rand(16, 1)
    state1 = state1/npl.norm(state1)
    mat1 = np.matmul(state1, np.transpose(state1))
    state2 = npr.rand(16, 1)
    state2 = state2/npl.norm(state2)
    mat2 = np.matmul(state2, np.transpose(state2))
    mat = (mat1 + mat2)/2.
    tl_mat = parafac(tl.tensor(mat), rank=2, normalize_factors=False)
    tl_vne = np.array(vonNeumann_entropy(tl_mat))
    dm_qutip = qutip.Qobj(mat)
    qutip_vne = qutip.entropy_vn(dm_qutip, base=2)
    assert np.allclose(tl_vne, qutip_vne)
