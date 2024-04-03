import pytest
import tensorly as tl

from ..testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)


def test_assert_allclose():
    tensor = tl.tensor([5, 5, 5], dtype=tl.float32)

    assert_allclose(tensor, tensor)
    assert_allclose(tensor, tensor + 1e-10)

    with pytest.raises(AssertionError):
        assert_allclose(tensor, tensor + 10)

    assert_allclose(tensor, tensor + 1, atol=2)
    with pytest.raises(AssertionError):
        assert_allclose(tensor, tensor + 1, atol=0.5)

    assert_allclose(tensor, tensor + 0.1 * tensor, rtol=0.2)
    with pytest.raises(AssertionError):
        assert_allclose(tensor, tensor + 0.1 * tensor, rtol=0.09)


def test_assert_equal():
    tensor = tl.tensor([5, 5, 5])

    assert_equal(tensor, tensor)
    assert_equal(tl.tensor([1, 2]), tl.tensor([1.0, 2.0]))

    assert_equal(tl.tensor([1, 1, 1]), tl.tensor(1))

    with pytest.raises(AssertionError):
        assert_equal(tensor, tensor + 10)


def test_assert_array_almost_equal():
    tensor = tl.tensor([5, 5, 5], dtype=tl.float32)

    assert_array_almost_equal(tensor, tensor)
    assert_array_almost_equal(tensor, tensor + 1e-10)

    with pytest.raises(AssertionError):
        assert_array_almost_equal(tensor, tensor + 10)

    decimal = 3
    assert_array_almost_equal(
        tensor, tensor + 1.5 * 10 ** (-decimal - 1), decimal=decimal
    )
    with pytest.raises(AssertionError):
        assert_array_almost_equal(
            tensor, tensor + 1.5 * 10 ** (-decimal), decimal=decimal
        )


def test_assert_array_equal():
    tensor = tl.tensor([5, 5, 5], dtype=tl.float32)

    assert_equal(tensor, tensor)
    assert_equal(tl.tensor([1, 2]), tl.tensor([1.0, 2.0]))

    with pytest.raises(AssertionError):
        assert_equal(tensor, tensor + 10)

    with pytest.raises(AssertionError):
        assert_array_equal(tl.tensor([1, 1, 1]), tl.tensor([1]))
