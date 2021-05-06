import tensorly as tl
from ...testing import assert_, assert_class_wrapper_correctly_passes_arguments

from .._symmetric_cp import symmetric_parafac_power_iteration, SymmetricCP


def test_symmetric_parafac_power_iteration(monkeypatch):
    """Test for symmetric Parafac optimized with robust tensor power iterations"""
    rng = tl.check_random_state(1234)
    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    
    size = 5
    rank = 4
    true_factor = tl.tensor(rng.random_sample((size, rank)))
    true_weights = tl.ones(rank)
    tensor = tl.cp_to_tensor((true_weights, [true_factor]*3))
    weights, factor = symmetric_parafac_power_iteration(tensor, rank=10, n_repeat=10, n_iteration=10)

    rec = tl.cp_to_tensor((weights, [factor]*3))
    error = tl.norm(rec - tensor, 2)
    error /= tl.norm(tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(tl.max(tl.abs(rec - tensor)) < tol_max_abs,
            'abs norm of reconstruction error higher than tol')
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, symmetric_parafac_power_iteration, SymmetricCP, ignore_args={}, rank=3)
