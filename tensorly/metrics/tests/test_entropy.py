import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import tensorly as tl
import qutip
from ..entropy import vonNeumann_entropy
from ...decomposition import parafac

mat_pure = tl.tensor([[3.53447351e-03, 3.30106160e-02, 1.88930958e-02, 6.31659996e-04,
        2.13221095e-02, 1.88003714e-02, 6.27753920e-03, 3.50348164e-02],
       [3.30106160e-02, 3.08306391e-01, 1.76454210e-01, 5.89946014e-03,
        1.99140259e-01, 1.75588200e-01, 5.86297890e-02, 3.27211639e-01],
       [1.88930958e-02, 1.76454210e-01, 1.00990733e-01, 3.37646124e-03,
        1.13974728e-01, 1.00495085e-01, 3.35558179e-02, 1.87274326e-01],
       [6.31659996e-04, 5.89946014e-03, 3.37646124e-03, 1.12886502e-04,
        3.81056006e-03, 3.35989008e-03, 1.12188431e-03, 6.26121314e-03],
       [2.13221095e-02, 1.99140259e-01, 1.13974728e-01, 3.81056006e-03,
        1.28628027e-01, 1.13415357e-01, 3.78699621e-02, 2.11351475e-01],
       [1.88003714e-02, 1.75588200e-01, 1.00495085e-01, 3.35989008e-03,
        1.13415357e-01, 1.00001871e-01, 3.33911310e-02, 1.86355211e-01],
       [6.27753920e-03, 5.86297890e-02, 3.35558179e-02, 1.12188431e-03,
        3.78699621e-02, 3.33911310e-02, 1.11494677e-02, 6.22249488e-02],
       [3.50348164e-02, 3.27211639e-01, 1.87274326e-01, 6.26121314e-03,
        2.11351475e-01, 1.86355211e-01, 6.22249488e-02, 3.47276152e-01]])

mat_mixed = tl.tensor([[0.05764554, 0.07271075, 0.05774785, 0.08440747, 0.01953676,
        0.02577351, 0.09599539, 0.07482428],
       [0.07271075, 0.2106877 , 0.13681119, 0.08753369, 0.1084978 ,
        0.10426328, 0.12271515, 0.2212482 ],
       [0.05774785, 0.13681119, 0.0922471 , 0.07437719, 0.06465954,
        0.06440067, 0.09704334, 0.14317333],
       [0.08440747, 0.08753369, 0.07437719, 0.12660652, 0.0152624 ,
        0.02632028, 0.14030151, 0.08937212],
       [0.01953676, 0.1084978 , 0.06465954, 0.0152624 , 0.06572383,
        0.05930842, 0.03368433, 0.11477836],
       [0.02577351, 0.10426328, 0.06440067, 0.02632028, 0.05930842,
        0.05479861, 0.04390419, 0.10996953],
       [0.09599539, 0.12271515, 0.09704334, 0.14030151, 0.03368433,
        0.04390419, 0.15988065, 0.12634305],
       [0.07482428, 0.2212482 , 0.14317333, 0.08937212, 0.11477836,
        0.10996953, 0.12634305, 0.23241005]])

def test_vonNeumann_entropy_pure_state():
    tl_vne = np.array(vonNeumann_entropy(mat_pure))
    assert np.allclose(tl_vne, np.array([0]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_pure_state_CP():
    mat = parafac(tl.tensor(mat_pure), rank=1, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(mat))
    assert np.allclose(tl_vne, np.array([0]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_mixed_state():
    tl_vne = np.array(vonNeumann_entropy(mat_mixed))
    assert np.allclose(tl_vne, np.array([0.7072100566602336]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_mixed_state_CP():
    mat = parafac(tl.tensor(mat_mixed), rank=2, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(mat))
    assert np.allclose(tl_vne, np.array([0.7072100566602336]), rtol=1e-03, atol=1e-03)

def test_vonNeumann_entropy_mixed_state_CP_unnormalized_factors():
    mat = parafac(tl.tensor(mat_mixed), rank=2, normalize_factors=True)
    tl_vne = np.array(vonNeumann_entropy(mat))
    assert np.allclose(tl_vne, np.array([0.7072100566602336]), rtol=1e-03, atol=1e-03)
