import itertools

import numpy as np
import pytest

import tensorly as tl
from ...random import random_parafac2
from ... import backend as T
from ...testing import assert_array_equal, assert_, assert_class_wrapper_correctly_passes_arguments
from .._parafac2 import Parafac2, parafac2, initialize_decomposition, _pad_by_zeros
from ...parafac2_tensor import Parafac2Tensor, parafac2_to_tensor, parafac2_to_slices
from ...metrics.factors import congruence_coefficient


@pytest.mark.parametrize(
    ("normalize_factors", "init"),
     itertools.product([True, False], ["random", "svd"])
)
def test_parafac2(monkeypatch, normalize_factors, init):
    rng = tl.check_random_state(1234)
    tol_norm_2 = 10e-2
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(25)],
        rank=rank,
        random_state=rng
    )
    # It is difficult to correctly identify B[i, :, r] if A[i, r] is small.
    # This is sensible, since then B[i, :, r] contributes little to the total value of X.
    # To test the PARAFAC2 decomposition in the precence of roundoff errors, we therefore add
    # 0.01 to the A factor matrix.
    random_parafac2_tensor.factors[0] = random_parafac2_tensor.factors[0] + 0.01

    tensor = parafac2_to_tensor(random_parafac2_tensor)
    slices = parafac2_to_slices(random_parafac2_tensor)

    rec, err = parafac2(
        slices,
        rank,
        random_state=rng,
        init=init,
        n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
        normalize_factors=normalize_factors,
        return_errors=True,
        n_iter_max=100
    )
    rec_tensor = parafac2_to_tensor(rec)

    error = T.norm(rec_tensor - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')

    # Test factor correlation
    A_sign = T.sign(random_parafac2_tensor.factors[0])
    rec_A_sign = T.sign(rec.factors[0])
    A_corr = congruence_coefficient(A_sign*random_parafac2_tensor.factors[0], rec_A_sign*rec.factors[0])[0]
    assert_(A_corr > 0.98)

    C_corr = congruence_coefficient(random_parafac2_tensor.factors[2], rec.factors[2])[0]
    assert_(C_corr > 0.98)

    for i, (true_proj, rec_proj) in enumerate(zip(random_parafac2_tensor.projections, rec.projections)):
        true_Bi = T.dot(true_proj, random_parafac2_tensor.factors[1])*A_sign[i]
        rec_Bi = T.dot(rec_proj, rec.factors[1])*rec_A_sign[i]
        Bi_corr = congruence_coefficient(true_Bi, rec_Bi)[0]
        assert_(Bi_corr > 0.98)
    
    # Test convergence criterion
    rec, err = parafac2(
        slices,
        rank,
        random_state=rng,
        init=init,
        normalize_factors=normalize_factors,
        tol=1e-10,
        absolute_tol=1e-4,
        return_errors=True
    )
    assert err[-1]**2 < 1e-4

    noisy_slices = [slice_ + tl.tensor(0.001*rng.standard_normal(T.shape(slice_))) for slice_ in slices]
    rec, err = parafac2(
        noisy_slices,
        rank,
        random_state=rng,
        init=init,
        normalize_factors=normalize_factors,
        tol=1e-4,
        absolute_tol=-1,
        return_errors=True,
        n_iter_max=1000
    )
    assert abs(err[-2]**2 - err[-1]**2) < (1e-4 * err[-2]**2)

    assert_class_wrapper_correctly_passes_arguments(monkeypatch, parafac2, Parafac2, ignore_args={'return_errors'}, rank=3)


def test_parafac2_nn():
    rng = tl.check_random_state(1234)
    tol_norm_2 = 1e-2
    rank = 3

    weights, factors, projections = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(25)],
        rank=rank,
        random_state=rng
    )
    factors = [tl.abs(factors[0]), factors[1], tl.abs(factors[2])]
    random_parafac2_tensor = Parafac2Tensor((weights, factors, projections))

    # It is difficult to correctly identify B[i, :, r] if A[i, r] is small.
    # This is sensible, since then B[i, :, r] contributes little to the total value of X.
    # To test the PARAFAC2 decomposition in the precence of roundoff errors, we therefore add
    # 0.01 to the A factor matrix.
    random_parafac2_tensor.factors[0] = random_parafac2_tensor.factors[0] + 0.01

    tensor = parafac2_to_tensor(random_parafac2_tensor)
    slices = parafac2_to_slices(random_parafac2_tensor)

    rec, err = parafac2(
        slices,
        rank,
        random_state=rng,
        init='random',
        nn_modes=[0, 2],
        n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
        normalize_factors=False,
        return_errors=True,
        n_iter_max=1000
    )
    rec_tensor = parafac2_to_tensor(rec)

    error = T.norm(rec_tensor - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')

    # Test factor correlation
    A_corr = congruence_coefficient(random_parafac2_tensor.factors[0], rec.factors[0])[0]
    assert_(A_corr > 0.98)

    C_corr = congruence_coefficient(random_parafac2_tensor.factors[2], rec.factors[2])[0]
    assert_(C_corr > 0.98)

    for i, (true_proj, rec_proj) in enumerate(zip(random_parafac2_tensor.projections, rec.projections)):
        true_Bi = T.dot(true_proj, random_parafac2_tensor.factors[1])
        rec_Bi = T.dot(rec_proj, rec.factors[1])
        Bi_corr = congruence_coefficient(true_Bi, rec_Bi)[0]
        assert_(Bi_corr > 0.98)
    
    # Fit with only one iteration to check non-negativity
    weights, factors, projections = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(25)],
        rank=rank,
        random_state=rng
    )
    # The default random parafac2 tensor has non-negative A and C
    # we therefore multiply them randomly with -1, 0 or 1 to get both positive and negative components
    factors = [factors[0] * T.tensor(rng.randint(-1, 2, factors[0].shape)),
               factors[1],
               factors[2] * T.tensor(rng.randint(-1, 2, factors[2].shape))]
    slices = parafac2_to_slices((weights, factors, projections))
    rec, err = parafac2(
        slices,
        rank,
        random_state=rng,
        init='random',
        nn_modes=[0, 2],
        n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
        normalize_factors=False,
        return_errors=True,
        n_iter_max=1
    )
    assert_(T.all(rec[1][0] > -1e-10))
    assert_(T.all(rec[1][2] > -1e-10))

    # Test that constraining B leads to a warning
    with pytest.warns(UserWarning):
        rec, err = parafac2(
            slices,
            rank,
            random_state=rng,
            init='random',
            nn_modes=[0, 1, 2],
            n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
            normalize_factors=False,
            return_errors=True,
            n_iter_max=1
        )
    with pytest.warns(UserWarning):
        rec, err = parafac2(
            slices,
            rank,
            random_state=rng,
            init='random',
            nn_modes='all',
            n_iter_parafac=2,  # Otherwise, the SVD init will converge too quickly
            normalize_factors=False,
            return_errors=True,
            n_iter_max=1
        )

def test_parafac2_slice_and_tensor_input():
    rank = 3

    random_parafac2_tensor = random_parafac2(
        shapes=[(15, 30) for _ in range(25)],
        rank=rank,
        random_state=1234
    )
    tensor = parafac2_to_tensor(random_parafac2_tensor)
    slices = parafac2_to_slices(random_parafac2_tensor)

    slice_rec = parafac2(slices, rank, random_state=1234, normalize_factors=False, n_iter_max=100)
    slice_rec_tensor = parafac2_to_tensor(slice_rec)

    tensor_rec = parafac2(tensor, rank, random_state=1234, normalize_factors=False, n_iter_max=100)
    tensor_rec_tensor = parafac2_to_tensor(tensor_rec)

    assert tl.max(tl.abs(slice_rec_tensor - tensor_rec_tensor)) < 1e-8


def test_parafac2_normalize_factors():
    rng = tl.check_random_state(1234)
    rank = 2  # Rank 2 so we only need to test rank of minimum and maximum

    random_parafac2_tensor = random_parafac2(
        shapes=[(15 + rng.randint(5), 30) for _ in range(25)],
        rank=rank,
        random_state=rng,
    )
    random_parafac2_tensor.factors[0] = random_parafac2_tensor.factors[0] + 0.1
    norms = tl.ones(rank)
    for factor in random_parafac2_tensor.factors:
        norms = norms*tl.norm(factor, axis=0)

    slices = parafac2_to_tensor(random_parafac2_tensor)

    unnormalized_rec = parafac2(slices, rank, random_state=rng, normalize_factors=False, n_iter_max=100)
    assert unnormalized_rec.weights[0] == 1

    normalized_rec = parafac2(slices, rank, random_state=rng, normalize_factors=True, n_iter_max=1000)
    assert tl.max(tl.abs(T.norm(normalized_rec.factors[0], axis=0) - 1)) < 1e-5
    assert abs(tl.max(norms) - tl.max(normalized_rec.weights))/tl.max(norms) < 1e-2
    assert abs(tl.min(norms) - tl.min(normalized_rec.weights))/tl.min(norms) < 1e-2

def test_parafac2_init_valid():
    rng = tl.check_random_state(1234)
    rank = 3

    random_parafac2_tensor = random_parafac2(shapes=[(15, 30)]*25, rank=rank, random_state=rng)
    tensor = parafac2_to_tensor(random_parafac2_tensor)
    weights, (A, B, C), projections = random_parafac2_tensor
    B = T.dot(projections[0], B)

    for init_method in ['random', 'svd', random_parafac2_tensor, (weights, (A, B, C))]:
        init = initialize_decomposition(tensor, rank, init=init_method)
        assert init.shape == random_parafac2_tensor.shape


def test_parafac2_init_error():
    rng = tl.check_random_state(1234)
    rank = 3

    random_parafac2_tensor = random_parafac2(shapes=[(15, 30)]*25, rank=rank, random_state=rng)
    tensor = parafac2_to_tensor(random_parafac2_tensor)

    with np.testing.assert_raises(ValueError):
        _ = initialize_decomposition(tensor, rank, init='bogus init type')

    with np.testing.assert_raises(ValueError):
        _ = initialize_decomposition(tensor, rank, init=('another', 'bogus', 'init', 'type'))

    rank = 4
    random_parafac2_tensor = random_parafac2(shapes=[(15, 3)]*25, rank=rank, random_state=rng)
    tensor = parafac2_to_tensor(random_parafac2_tensor)

    with pytest.raises(Exception):
        _ = initialize_decomposition(tensor, rank, init='svd')


def test_parafac2_to_tensor():
    rng = tl.check_random_state(1234)
    rank = 3

    I = 25
    J = 15
    K = 30

    weights, factors, projections = random_parafac2(shapes=[(J, K)]*I, rank=rank, random_state=rng)

    constructed_tensor = parafac2_to_tensor((weights, factors, projections))
    tensor_manual = T.zeros((I, J, K), **T.context(weights))

    for i in range(I):
        Bi = T.dot(projections[i], factors[1])
        for j in range(J):
            for k in range(K):
                for r in range(rank):
                    tensor_manual = tl.index_update(tensor_manual, tl.index[i, j, k],  tensor_manual[i, j, k] + factors[0][i][r]*Bi[j][r]*factors[2][k][r])

    assert_(tl.max(tl.abs(constructed_tensor - tensor_manual)) < 1e-6)


def test_pad_by_zeros():
    """Test that if we pad a tensor by zeros, then it doesn't change.

    This failed for TensorFlow at some point.
    """
    rng = tl.check_random_state(1234)
    rank = 3

    I = 25
    J = 15
    K = 30

    weights, factors, projections = random_parafac2(shapes=[(J, K)]*I, rank=rank, random_state=rng)
    constructed_tensor = parafac2_to_tensor((weights, factors, projections))
    padded_tensor = _pad_by_zeros(constructed_tensor)
    assert_(tl.max(tl.abs(constructed_tensor - padded_tensor)) < 1e-10)
