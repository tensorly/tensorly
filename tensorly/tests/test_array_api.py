import ast
import os
import tempfile

import numpy as np
import pytest

import tensorly as tl
from .. import backend as T
from ..testing import assert_array_equal, assert_array_almost_equal


def _to_np(x):
    return tl.to_numpy(x)


def test_argsort():
    x_np = np.array([[1.0, 2.0, 2.0, 1.0], [3.0, 0.0, 3.0, 0.0]])
    x = tl.tensor(x_np)
    for axis in (0, 1, -1):
        assert_array_equal(
            _to_np(tl.argsort(x, axis=axis)), np.argsort(x_np, axis=axis)
        )
    # Default axis is the last one
    assert_array_equal(_to_np(tl.argsort(x)), np.argsort(x_np, axis=-1))
    # Descending order, stable: ties keep their original relative order
    res = _to_np(tl.argsort(x, axis=1, descending=True))
    assert_array_equal(res, np.array([[1, 2, 0, 3], [0, 2, 1, 3]]))


def test_sort():
    x_np = np.array([[1.0, 5.0, 2.0], [3.0, 0.0, 4.0]])
    x = tl.tensor(x_np)
    assert_array_equal(_to_np(tl.sort(x)), np.sort(x_np, axis=-1))
    assert_array_equal(_to_np(tl.sort(x, axis=0)), np.sort(x_np, axis=0))
    assert_array_equal(
        _to_np(tl.sort(x, axis=1, descending=True)), np.flip(np.sort(x_np, axis=1), 1)
    )


@pytest.mark.parametrize("args", [(3,), (3, 4), (4, 3)])
@pytest.mark.parametrize("k", [-1, 0, 2])
def test_eye(args, k):
    assert_array_equal(_to_np(tl.eye(*args, k=k)), np.eye(*args, k=k))


def test_trace():
    x_np = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    x = tl.tensor(x_np)
    # The trace is taken over the last two axes
    for offset in (0, 1, -1):
        expected = np.trace(x_np, offset=offset, axis1=-2, axis2=-1)
        assert_array_almost_equal(tl.trace(x, offset=offset), expected)
    matrix = tl.tensor(np.arange(9, dtype=float).reshape(3, 3))
    assert_array_almost_equal(tl.trace(matrix), 12.0)


def test_tensordot():
    a_np = np.random.RandomState(0).rand(2, 3, 4)
    b_np = np.random.RandomState(1).rand(3, 4, 5)
    a, b = tl.tensor(a_np), tl.tensor(b_np)
    for axes in (0, 1, ([1, 2], [0, 1])):
        a_ = a if axes != 0 else a
        if axes == 1:
            b_np_, b_ = np.random.RandomState(1).rand(4, 5), None
            b_ = tl.tensor(b_np_)
            expected = np.tensordot(a_np, b_np_, axes=1)
            res = tl.tensordot(a, b_, axes=1)
        else:
            expected = np.tensordot(a_np, b_np, axes=axes)
            res = tl.tensordot(a, b, axes=axes)
        assert_array_almost_equal(res, expected)


@pytest.mark.parametrize("name", ["sum", "prod", "mean", "max", "min", "all", "any"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1), (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_reductions(name, axis, keepdims):
    x_np = np.random.RandomState(0).rand(2, 3, 4) + 0.5
    if name in ("all", "any"):
        x_np = x_np > 1.0
    x = tl.tensor(x_np)
    expected = getattr(np, name)(x_np, axis=axis, keepdims=keepdims)
    res = getattr(tl, name)(x, axis=axis, keepdims=keepdims)
    assert tuple(res.shape) == np.shape(expected)
    assert_array_almost_equal(_to_np(res).astype(float), expected.astype(float))


def test_reshape():
    x = tl.tensor(np.arange(12, dtype=float))
    assert tuple(tl.reshape(x, (3, 4)).shape) == (3, 4)
    assert tuple(tl.reshape(x, (-1,)).shape) == (12,)


def _stub_path():
    return os.path.join(os.path.dirname(T.__file__), "__init__.pyi")


def test_stub_matches_backend_namespace():
    """The type stub must declare exactly the functions and attributes of the backend"""
    with open(_stub_path()) as f:
        tree = ast.parse(f.read())
    declared = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            declared.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared.add(node.target.id)
    backend_names = set(T.BackendManager._functions) | set(T.BackendManager._attributes)
    assert backend_names <= declared
    # The standard functions must be callable on the current backend
    for name in declared & backend_names:
        assert hasattr(tl, name), name


def test_removed_non_standard_names():
    for name in ("concatenate", "transpose", "dot", "arcsin", "arccos", "arctan"):
        assert not hasattr(tl, name), name


VALID_CALLS = """
import tensorly as tl
x = tl.zeros((2, 3))
tl.concat([x, x], axis=1)
tl.permute_dims(x, (1, 0))
tl.clip(x, min=0.0, max=1.0)
tl.sort(x, axis=0, descending=True)
tl.argsort(x, descending=True)
tl.sum(x, axis=(0, 1), keepdims=True)
tl.tensordot(x, x, axes=1)
tl.eye(3, 4, k=1)
tl.trace(x, offset=1)
tl.matmul(x, x)
tl.reshape(x, (3, 2))
"""

INVALID_CALLS = [
    "tl.concat([x, x], 1)",  # axis is keyword-only
    "tl.clip(x, a_min=0.0)",  # renamed to min
    "tl.sort(x, 0)",  # axis is keyword-only
    "tl.sum(x, 0)",  # axis is keyword-only
    "tl.tensordot(x, x, 1)",  # axes is keyword-only
    "tl.trace(x, 1)",  # offset is keyword-only
    "tl.permute_dims(x)",  # axes is required
    "tl.eye(3, 3, 1)",  # k is keyword-only
    "tl.where(x)",  # the one-argument form is not in the standard
    "tl.transpose(x)",  # not in the standard
    "tl.concatenate([x, x])",  # not in the standard
    "tl.dot(x, x)",  # not in the standard
]


def _run_mypy(code):
    api = pytest.importorskip("mypy.api")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snippet.py")
        with open(path, "w") as f:
            f.write(code)
        out, _, _ = api.run(
            [
                "--ignore-missing-imports",
                "--follow-imports=silent",
                "--no-incremental",
                "--cache-dir=" + os.path.join(tmp, "cache"),
                "--check-untyped-defs",
                path,
            ]
        )
    return [
        line
        for line in out.splitlines()
        if line.startswith(path) and ": error:" in line
    ]


def test_type_stub_accepts_standard_calls():
    assert _run_mypy(VALID_CALLS) == []


def test_type_stub_rejects_non_standard_calls():
    header = "import tensorly as tl\nx = tl.zeros((2, 3))\n"
    code = header + "\n".join(INVALID_CALLS) + "\n"
    errors = _run_mypy(code)
    flagged = {int(line.split(":")[1]) - 3 for line in errors}
    assert flagged == set(range(len(INVALID_CALLS))), errors
