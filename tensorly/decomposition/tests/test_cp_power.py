import tensorly as tl
from ...random import random_cp
from ...testing import assert_, assert_class_wrapper_correctly_passes_arguments

from .._cp_power import parafac_power_iteration, CPPower


def test_parafac_power_iteration(monkeypatch):
    """Test for symmetric Parafac optimized with robust tensor power iterations"""
    rng = tl.check_random_state(1234)
    tol_norm_2 = 10e-1
    tol_max_abs = 10e-1
    
    shape = (5, 3, 4)
    rank = 4
    tensor = random_cp(shape, rank=rank, full=True, random_state=rng)
    ktensor = parafac_power_iteration(tensor, rank=10, n_repeat=10, n_iteration=10)

    rec = tl.cp_to_tensor(ktensor)
    error = tl.norm(rec - tensor, 2)/tl.norm(tensor, 2)
    assert_(error < tol_norm_2,
            f'Norm 2 of reconstruction error={error} higher than tol={tol_norm_2}.')
    error = tl.max(tl.abs(rec - tensor))
    assert_(error < tol_max_abs,
            f'Absolute norm of reconstruction error={error} higher than tol={tol_max_abs}.')

    
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, parafac_power_iteration, CPPower, ignore_args={}, rank=3)
