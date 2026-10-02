from concurrent.futures import ThreadPoolExecutor

import pytest
import tensorly as tl

from tensorly.testing import assert_array_almost_equal


@pytest.fixture(autouse=True)
def restore_tenalg_backend():
    original = tl.tenalg.get_backend()
    try:
        yield
    finally:
        tl.tenalg.set_backend(original)


@pytest.mark.parametrize("initial", ["core", "einsum"])
@pytest.mark.parametrize("temporary", ["core", "einsum"])
@pytest.mark.parametrize("as_instance", [False, True])
@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("local_threadsafe", [False, True])
def test_tenalg_backend_context(
    initial, temporary, as_instance, body_error, local_threadsafe
):
    tl.tenalg.set_backend(initial)
    original = tl.tenalg.current_backend()
    backend = tl.tenalg.load_backend(temporary) if as_instance else temporary

    def use_backend():
        with tl.tenalg.backend_context(backend, local_threadsafe=local_threadsafe):
            assert tl.tenalg.get_backend() == temporary
            with ThreadPoolExecutor(max_workers=1) as executor:
                worker_backend = executor.submit(tl.tenalg.get_backend).result()
            assert worker_backend == (initial if local_threadsafe else temporary)
            tensor = tl.tensor([1.0, 2.0, 3.0])
            assert_array_almost_equal(tl.tenalg.inner(tensor, tensor), 14)
            if body_error:
                raise RuntimeError("body sentinel")

    if body_error:
        with pytest.raises(RuntimeError, match="body sentinel"):
            use_backend()
    else:
        use_backend()
    assert tl.tenalg.current_backend() is original
    assert tl.tenalg.get_backend() == initial


@pytest.mark.parametrize("backend_name", ["core", "einsum"])
def test_tenalg_backend_instance_identity(backend_name):
    backend = tl.tenalg.load_backend(backend_name)
    tl.tenalg.set_backend(backend)
    assert tl.tenalg.current_backend() is backend
    other = "einsum" if backend_name == "core" else "core"
    with tl.tenalg.backend_context(other):
        assert tl.tenalg.get_backend() == other
    assert tl.tenalg.current_backend() is backend


@pytest.mark.parametrize("initial", ["core", "einsum"])
def test_nested_tenalg_backend_context(initial):
    tl.tenalg.set_backend(initial)
    original = tl.tenalg.current_backend()
    other = "einsum" if initial == "core" else "core"
    with tl.tenalg.backend_context(other):
        outer = tl.tenalg.current_backend()
        with tl.tenalg.backend_context(initial):
            assert tl.tenalg.get_backend() == initial
        assert tl.tenalg.current_backend() is outer
    assert tl.tenalg.current_backend() is original


def test_invalid_tenalg_backend_preserves_state():
    original = tl.tenalg.current_backend()
    with pytest.raises(ValueError, match="Unknown backend name"):
        tl.tenalg.set_backend("not-a-tenalg-backend")
    assert tl.tenalg.current_backend() is original


def test_numeric_backend_instance_context():
    original = tl.backend.current_backend()
    with tl.backend_context(original):
        assert tl.backend.current_backend() is original
        assert_array_almost_equal(tl.sum(tl.tensor([1.0, 2.0, 3.0])), 6)
    assert tl.backend.current_backend() is original
