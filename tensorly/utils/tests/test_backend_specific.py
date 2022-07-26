import pytest

import tensorly.utils.backend_specific
from tensorly.utils.backend_specific import register_backend_specific


@pytest.fixture
def monkeypatch_available_backends(monkeypatch):
    monkeypatch.setattr(
        tensorly.utils.backend_specific, "available_backend_names", ["backend"]
    )


def test_register_backend_specific_fails_with_added_arguments(
    monkeypatch_available_backends,
):
    def f_backend_specific(x, y):
        return x

    with pytest.raises(TypeError):

        @register_backend_specific("backend", f_backend_specific)
        def f(x):
            return x


def test_register_backend_specific_fails_with_missing_arguments(
    monkeypatch_available_backends,
):
    def f_backend_specific(x):
        return x

    with pytest.raises(TypeError):

        @register_backend_specific("backend", f_backend_specific)
        def f(x, y):
            return x


def test_register_backend_specific_fails_with_different_default_value(
    monkeypatch_available_backends,
):
    def f_backend_specific(x=1):
        return x

    with pytest.raises(TypeError):

        @register_backend_specific("backend", f_backend_specific)
        def f(x=0):
            return x


def test_register_backend_specific_fails_with_missing_default_value(
    monkeypatch_available_backends,
):
    def f_backend_specific(x):
        return x

    with pytest.raises(TypeError):

        @register_backend_specific("backend", f_backend_specific)
        def f(x=0):
            return x


def test_register_backend_specific_fails_with_added_default_value(
    monkeypatch_available_backends,
):
    def f_backend_specific(x=1):
        return x

    with pytest.raises(TypeError):

        @register_backend_specific("backend", f_backend_specific)
        def f(x):
            return x


def test_register_backend_specific_uses_correct_function(monkeypatch):
    def f_backend_specific():
        return 1

    monkeypatch.setattr(
        tensorly.utils.backend_specific, "available_backend_names", ["backend"]
    )

    @register_backend_specific("backend", f_backend_specific)
    def f():
        return 2

    # Check if the default function is called when the backend is not "backend"
    assert f() == 2

    # Monkeypatch get_backend and check that the specific function is called when backend is "backend"
    def get_backend():
        return "backend"

    monkeypatch.setattr(tensorly.utils.backend_specific, "get_backend", get_backend)
    assert f() == 1


def test_register_backend_specific_fails_with_unavailable_backend():
    def f_backend_specific(x):
        return x

    with pytest.raises(ValueError):

        @register_backend_specific("unavailable_backend", f_backend_specific)
        def f(x):
            return x
