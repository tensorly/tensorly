import numpy as np
import pytest

import tensorly as tl
from tensorly.tenalg import backend_context, unfolding_dot_khatri_rao
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao_memory


@pytest.mark.parametrize("implementation", ["core", "einsum", "memory"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("weight_kind", ["complex", "real", "implicit"])
def test_mttkrp_weights(implementation, mode, dtype, weight_kind):
    values = np.arange(24).reshape(2, 3, 4) / 8 + 0.5j
    factor_values = [
        np.arange(size * 2).reshape(size, 2) / 4 + 1 + 0.25j for size in values.shape
    ]
    weight_values = {
        "complex": np.array([1 + 2j, 2 - 1j]),
        "real": np.array([2, -1]),
        "implicit": np.ones(2),
    }[weight_kind]
    # Compute the Hermitian contraction directly, without TensorLy unfolding
    # or Khatri-Rao products, including conjugation of each component's weight.
    equations = ["abc,br,cr,r->ar", "abc,ar,cr,r->br", "abc,ar,br,r->cr"]
    expected = np.einsum(
        equations[mode],
        values,
        *[factor.conj() for i, factor in enumerate(factor_values) if i != mode],
        weight_values.conj(),
    )
    context = {"dtype": getattr(tl, dtype)}
    tensor = tl.tensor(values, **context)
    factors = [tl.tensor(factor, **context) for factor in factor_values]
    weights = None if weight_kind == "implicit" else tl.tensor(weight_values, **context)

    if implementation == "memory":
        actual = unfolding_dot_khatri_rao_memory(tensor, (weights, factors), mode)
    else:
        with backend_context(implementation):
            actual = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)

    assert tl.shape(actual) == expected.shape
    assert tl.context(actual)["dtype"] == context["dtype"]
    np.testing.assert_allclose(tl.to_numpy(actual), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(tl.to_numpy(tensor), values)
    for factor, original in zip(factors, factor_values):
        np.testing.assert_array_equal(tl.to_numpy(factor), original)
    if weights is not None:
        np.testing.assert_array_equal(tl.to_numpy(weights), weight_values)
