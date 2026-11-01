import math
import pytest
import tensorly as tl
from tensorly.cp_tensor import CPTensor, cp_to_tensor
from ..entropy import vonneumann_entropy
from ..entropy import tt_vonneumann_entropy, cp_vonneumann_entropy
from ...decomposition import parafac, tensor_train
from tensorly.testing import assert_array_almost_equal


def test_vonneumann_entropy_pure_state():
    """Test for vonneumann_entropy on 2-dimensional tensors.
    This test checks that pure states have a VNE of zero.
    """
    state = tl.randn((8, 1))
    state = state / tl.norm(state)
    mat_pure = tl.dot(state, tl.transpose(state))
    tl_vne = vonneumann_entropy(mat_pure)
    assert_array_almost_equal(tl_vne, 0, decimal=3)


def test_tt_vonneumann_entropy_pure_state():
    """Test for tt_vonneumann_entropy TT tensors.
    This test checks that pure states have a VNE of zero.
    """
    state = tl.randn((8, 1))
    state = state / tl.norm(state)
    mat_pure = tl.reshape(tl.dot(state, tl.transpose(state)), (2, 2, 2, 2, 2, 2))
    mat_pure = tensor_train(mat_pure, rank=(1, 3, 2, 1, 2, 3, 1))
    tl_vne = tt_vonneumann_entropy(mat_pure)
    assert_array_almost_equal(tl_vne, 0, decimal=3)


def test_cp_vonneumann_entropy_pure_state():
    """Test for cp_vonneumann_entropy on 2-dimensional CP tensors.
    This test checks that pure states have a VNE of zero.
    """
    state = tl.randn((8, 1))
    state = state / tl.norm(state)
    mat_pure = tl.dot(state, tl.transpose(state))
    mat = parafac(mat_pure, rank=1, normalize_factors=True)
    tl_vne = cp_vonneumann_entropy(mat)
    assert_array_almost_equal(tl_vne, 0, decimal=3)


def test_vonneumann_entropy_mixed_state():
    """Test for vonneumann_entropy on 2-dimensional tensors.
    This test checks that the VNE of mixed states is calculated correctly.
    """
    state1 = tl.tensor(
        [
            [
                0.03004805,
                0.42426117,
                0.5483771,
                0.4784077,
                0.25792725,
                0.34388784,
                0.99927586,
                0.96605812,
            ]
        ]
    )
    state1 = state1 / tl.norm(state1)
    state2 = tl.tensor(
        [
            [
                0.84250089,
                0.43429687,
                0.26551928,
                0.18262211,
                0.55584835,
                0.2565509,
                0.33197401,
                0.97741178,
            ]
        ]
    )
    state2 = state2 / tl.norm(state2)
    mat_mixed = tl.tensor(
        (tl.dot(tl.transpose(state1), state1) + tl.dot(tl.transpose(state2), state2))
        / 2.0
    )
    tensor_mixed = tl.reshape(mat_mixed, (4, 2, 4, 2))
    actual_vne = 0.5546
    tl_vne = vonneumann_entropy(mat_mixed)
    tl_tensor_vne = vonneumann_entropy(tensor_mixed)
    assert_array_almost_equal(tl_vne, actual_vne, decimal=3)
    assert_array_almost_equal(tl_tensor_vne, actual_vne, decimal=3)


def test_tt_vonneumann_entropy_mixed_state():
    """Test for tt_vonneumann_entropy on TT tensors.
    This test checks that the VNE of mixed states is calculated correctly.
    """
    state1 = tl.tensor(
        [
            [
                0.03004805,
                0.42426117,
                0.5483771,
                0.4784077,
                0.25792725,
                0.34388784,
                0.99927586,
                0.96605812,
            ]
        ]
    )
    state1 = state1 / tl.norm(state1)
    state2 = tl.tensor(
        [
            [
                0.84250089,
                0.43429687,
                0.26551928,
                0.18262211,
                0.55584835,
                0.2565509,
                0.33197401,
                0.97741178,
            ]
        ]
    )
    state2 = state2 / tl.norm(state2)
    mat_mixed = tl.tensor(
        (tl.dot(tl.transpose(state1), state1) + tl.dot(tl.transpose(state2), state2))
        / 2.0
    )
    actual_vne = 0.5546
    tt_mixed = tl.reshape(mat_mixed, (2, 2, 2, 2, 2, 2))
    tt_mixed = tensor_train(tt_mixed, rank=[1, 2, 4, 8, 4, 2, 1])
    tl_vne = tt_vonneumann_entropy(tt_mixed)
    assert_array_almost_equal(tl_vne, actual_vne, decimal=3)


# Overfactoring causes a singular matrix error.
@pytest.mark.xfail()
def test_cp_vonneumann_entropy_mixed_state():
    """Test for cp_vonneumann_entropy on CP tensors.
    This test checks that the VNE of mixed states is calculated correctly.
    """
    state1 = tl.tensor(
        [
            [
                0.03004805,
                0.42426117,
                0.5483771,
                0.4784077,
                0.25792725,
                0.34388784,
                0.99927586,
                0.96605812,
            ]
        ]
    )
    state1 = state1 / tl.norm(state1)
    state2 = tl.tensor(
        [
            [
                0.84250089,
                0.43429687,
                0.26551928,
                0.18262211,
                0.55584835,
                0.2565509,
                0.33197401,
                0.97741178,
            ]
        ]
    )
    state2 = state2 / tl.norm(state2)
    mat_mixed = tl.tensor(
        (tl.dot(tl.transpose(state1), state1) + tl.dot(tl.transpose(state2), state2))
        / 2.0
    )
    actual_vne = 0.5546
    mat = parafac(mat_mixed, rank=2, normalize_factors=True)
    mat_unnorm = parafac(mat_mixed, rank=2, normalize_factors=False)
    tl_vne = cp_vonneumann_entropy(mat)
    tl_vne_unnorm = cp_vonneumann_entropy(mat_unnorm)
    assert_array_almost_equal(tl_vne, actual_vne, decimal=3)
    assert_array_almost_equal(tl_vne_unnorm, actual_vne, decimal=3)


def binary_entropy(p):
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize(
    "case",
    [
        "nonorthogonal",
        "different_factors_none_weights",
        "duplicate_pure",
        "orthogonal",
        "rank_one",
        "higher_order",
    ],
)
def test_cp_entropy_public_density(case, dtype_name):
    dtype = getattr(tl, dtype_name)

    def tensor(x):
        return tl.tensor(x, dtype=dtype)

    a = [[1, 1 / math.sqrt(2)], [0, 1 / math.sqrt(2)]]
    weights = tensor([0.5, 0.5])
    if case == "nonorthogonal":
        factors = [tensor(a), tensor(a)]
        expected = binary_entropy((1 + 1 / math.sqrt(2)) / 2)
        expected_dense = tensor([[0.75, 0.25], [0.25, 0.25]])
    elif case == "different_factors_none_weights":
        weights = None
        factors = [tensor([[1, 1], [0, 1]]), tensor([[0.5, 0], [-0.5, 0.5]])]
        expected = 1.0
        expected_dense = tensor([[0.5, 0], [0, 0.5]])
    elif case == "duplicate_pure":
        factors = [tensor([[1, 1], [0, 0]]) for _ in range(2)]
        expected = 0.0
        expected_dense = tensor([[1, 0], [0, 0]])
    elif case == "orthogonal":
        factors = [tensor([[1, 0], [0, 1]]) for _ in range(2)]
        expected = 1.0
        expected_dense = tensor([[0.5, 0], [0, 0.5]])
    elif case == "rank_one":
        weights = tensor([1.0])
        factors = [tensor([[1.0], [0.0]]) for _ in range(2)]
        expected = 0.0
        expected_dense = tensor([[1, 0], [0, 0]])
    else:
        factors = [tensor(a) for _ in range(4)]
        expected = binary_entropy(0.75)
        expected_dense = tensor(
            [
                [0.625, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125],
            ]
        )
    cp = CPTensor((weights, factors))
    dense = cp_to_tensor(cp)
    assert (
        float(tl.max(tl.abs(tl.reshape(dense, expected_dense.shape) - expected_dense)))
        < 2e-6
    )
    dense_entropy = vonneumann_entropy(dense)
    assert float(dense_entropy) == pytest.approx(expected, abs=2e-6)
    actual = cp_vonneumann_entropy(cp)
    assert actual.dtype == factors[0].dtype
    assert tl.shape(actual) == ()
    assert tl.context(actual).get("device") == tl.context(factors[0]).get("device")
    assert float(actual) == pytest.approx(expected, abs=2e-6)


@pytest.mark.skipif(
    tl.get_backend() != "pytorch", reason="PyTorch-specific autograd check"
)
def test_smooth_nonorthogonal_factor_gradient():
    import torch

    angle = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    factor = torch.stack(
        [torch.stack([one, torch.cos(angle)]), torch.stack([zero, torch.sin(angle)])]
    )
    cp = CPTensor((torch.tensor([0.5, 0.5], dtype=torch.float64), [factor, factor]))
    dense = vonneumann_entropy(cp_to_tensor(cp))
    (expected_grad,) = torch.autograd.grad(dense, angle, retain_graph=True)
    actual = cp_vonneumann_entropy(cp)
    assert actual.requires_grad
    (actual_grad,) = torch.autograd.grad(actual, angle)
    assert torch.isfinite(expected_grad) and abs(expected_grad.item()) > 0.01
    assert torch.isfinite(actual_grad)
    assert actual_grad.item() == pytest.approx(expected_grad.item(), abs=1e-10)
