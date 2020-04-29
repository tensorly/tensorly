import itertools

import numpy as np
import pytest

import tensorly as tl
from ...random import check_random_state, random_parafac2
from ... import backend as T
from ...testing import assert_array_equal, assert_
from ..parafac2 import parafac2, initialize_decomposition
from ...parafac2_tensor import parafac2_to_tensor, parafac2_to_slices


def corrcoef(A, B):
    Ac = A - T.mean(A, axis=0)
    Bc = B - T.mean(B, axis=0)

    As = Ac/T.norm(Ac, axis=0)
    Bs = Bc/T.norm(Bc, axis=0)

    return T.dot(T.transpose(As), Bs)


def best_correlation(A, B):
    _, r = T.shape(A)
    corr_matrix = T.abs(corrcoef(A, B))

    best_corr = 0
    for permutation in itertools.permutations(range(r)):
        corr = 1
        for i, j in zip(range(r), permutation):
            corr *= corr_matrix[i, j]
        
        if corr > best_corr:
            best_corr = corr
    
    return best_corr

@pytest.mark.parametrize("normalize_factors", [False])
def test_parafac2(normalize_factors):
    rng = check_random_state(1234)
    tol_norm_2 = 10e-2
    rank = 6

    random_parafac2_tensor = random_parafac2(
        shapes=[(50 + rng.randint(10),60) for _ in range(40)],
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

    rec = parafac2(slices, rank, random_state=rng, normalize_factors=normalize_factors)
    rec_tensor = parafac2_to_tensor(rec)

    error = T.norm(rec_tensor - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            'norm 2 of reconstruction higher than tol')

    # Test factor correlation
    A_sign = T.sign(random_parafac2_tensor.factors[0])
    rec_A_sign = T.sign(rec.factors[0])
    A_corr = best_correlation(A_sign*random_parafac2_tensor.factors[0], rec_A_sign*rec.factors[0])
    assert_(T.prod(A_corr) > 0.98**rank)

    C_corr = best_correlation(random_parafac2_tensor.factors[2], rec.factors[2])
    assert_(T.prod(C_corr) > 0.98**rank)

    for i, (true_proj, rec_proj) in enumerate(zip(random_parafac2_tensor.projections, rec.projections)):
        true_Bi = T.dot(true_proj, random_parafac2_tensor.factors[1])*A_sign[i]
        rec_Bi = T.dot(rec_proj, rec.factors[1])*rec_A_sign[i]
        Bi_corr = best_correlation(true_Bi, rec_Bi)
        assert_(T.prod(Bi_corr) > 0.98**rank)


def test_parafac2_slice_and_tensor_input():
    rank = 6

    random_parafac2_tensor = random_parafac2(
        shapes=[(50, 60) for _ in range(40)],
        rank=rank,
        random_state=1234
    )
    tensor = parafac2_to_tensor(random_parafac2_tensor)
    slices = parafac2_to_slices(random_parafac2_tensor)

    slice_rec = parafac2(slices, rank, random_state=1234, normalize_factors=False)
    slice_rec_tensor = parafac2_to_tensor(slice_rec)

    tensor_rec = parafac2(tensor, rank, random_state=1234, normalize_factors=False)
    tensor_rec_tensor = parafac2_to_tensor(tensor_rec)

    assert tl.max(tl.abs(slice_rec_tensor - tensor_rec_tensor)) < 1e-8


def test_parafac2_normalize_factors():
    rng = check_random_state(1234)
    rank = 6

    random_parafac2_tensor = random_parafac2(
        shapes=[(50 + rng.randint(10),60) for _ in range(40)],
        rank=rank,
        random_state=rng,
    )
    tl.index_update(random_parafac2_tensor.weights, tl.index[0], 100)
    slices = parafac2_to_tensor(random_parafac2_tensor)
    normalized_rec = parafac2(slices, rank, random_state=rng, normalize_factors=True)
    assert normalized_rec.weights[0] > 1
    assert tl.max(tl.abs(T.norm(normalized_rec.factors[0], axis=0) - 1)) < 1e-5
    
    unnormalized_rec = parafac2(slices, rank, random_state=rng, normalize_factors=False)
    assert unnormalized_rec.weights[0] == 1


def test_parafac2_init_valid():
    rng = check_random_state(1234)
    rank = 6

    random_parafac2_tensor = random_parafac2(shapes=[(30,60)]*50, rank=rank, random_state=rng)
    tensor = parafac2_to_tensor(random_parafac2_tensor)
    weights, (A, B, C), projections = random_parafac2_tensor
    B = T.dot(projections[0], B)
    
    for init_method in ['random', 'svd', random_parafac2_tensor, (weights, (A, B, C))]:
        init = initialize_decomposition(tensor, rank, init=init_method)
        assert init.shape == random_parafac2_tensor.shape


def test_parafac2_init_error():
    rng = check_random_state(1234)
    rank = 6

    random_parafac2_tensor = random_parafac2(shapes=[(30,60)]*50, rank=rank, random_state=rng)
    tensor = parafac2_to_tensor(random_parafac2_tensor)

    with np.testing.assert_raises(ValueError):
        _ = initialize_decomposition(tensor, rank, init='bogus init type')

    with np.testing.assert_raises(ValueError):
        _ = initialize_decomposition(tensor, rank, init=('another', 'bogus', 'init', 'type'))

def test_parafac2_to_tensor():
    rng = check_random_state(1234)
    rank = 6

    I = 50
    J = 30
    K = 60

    weights, factors, projections = random_parafac2(shapes=[(J, K)]*I, rank=rank, random_state=rng)

    constructed_tensor = parafac2_to_tensor((weights, factors, projections))
    tensor_manual = T.zeros((I, J, K), **T.context(weights))

    for i in range(I):
        Bi = T.dot(projections[i], factors[1])
        for j in range(J):
            for k in range(K):
                for r in range(rank):
                    tl.index_update(tensor_manual, tl.index[i, j, k],  tensor_manual[i, j, k] + factors[0][i][r]*Bi[j][r]*factors[2][k][r])

    assert_(tl.max(tl.abs(constructed_tensor - tensor_manual)) < 1e-6)
    
    
