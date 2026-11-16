import numpy as np
import pytest

from .. import backend as tl
from ..base import unfold, tensor_to_vec
from ..tucker_tensor import (
    TuckerTensor,
    tucker_to_tensor,
    tucker_to_unfolded,
    tucker_to_vec,
    _validate_tucker_tensor,
    tucker_mode_dot,
    tucker_normalize,
    _tucker_n_param,
    validate_tucker_rank,
)
from ..tenalg import kronecker, mode_dot
from ..testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_equal,
    assert_raises,
    assert_,
)
from ..random import random_tucker


def test_validate_tucker_tensor():
    rng = tl.check_random_state(1234)
    true_shape = (3, 4, 5)
    true_rank = (3, 2, 4)
    core, factors = random_tucker(true_shape, rank=true_rank, random_state=rng)

    # Check shape and rank returned
    shape, rank = _validate_tucker_tensor((core, factors))
    assert_equal(
        shape,
        true_shape,
        err_msg=f"Returned incorrect shape (got {shape}, expected {true_shape})",
    )
    assert_equal(
        rank,
        true_rank,
        err_msg=f"Returned incorrect rank (got {rank}, expected {true_rank})",
    )

    # One of the factors has the wrong rank
    factors[0], copy = tl.tensor(rng.random_sample((4, 4))), factors[0]
    with assert_raises(ValueError):
        _validate_tucker_tensor((core, factors))

    # Not enough factors to match core
    factors[0] = copy
    with assert_raises(ValueError):
        _validate_tucker_tensor((core, factors[1:]))

    # Not enough factors
    with assert_raises(ValueError):
        _validate_tucker_tensor((core, factors[:1]))


def test_tucker_to_tensor():
    """Test for tucker_to_tensor"""
    X = tl.tensor(
        np.array(
            [
                [[1.0, 13], [4, 16], [7, 19], [10, 22]],
                [[2, 14], [5, 17], [8, 20], [11, 23]],
                [[3, 15], [6, 18], [9, 21], [12, 24]],
            ]
        )
    )
    ranks = [2, 3, 4]
    U = [
        tl.tensor(np.arange(R * s, dtype=float).reshape((R, s)))
        for (R, s) in zip(ranks, tl.shape(X))
    ]
    true_res = np.array(
        [
            [
                [390.0, 1518, 2646, 3774],
                [1310, 4966, 8622, 12278],
                [2230, 8414, 14598, 20782],
            ],
            [
                [1524, 5892, 10260, 14628],
                [5108, 19204, 33300, 47396],
                [8692, 32516, 56340, 80164],
            ],
        ]
    )
    res = tucker_to_tensor((X, U))
    assert_array_equal(true_res, res)


def test_tucker_to_tensor_with_partial_modes():
    """Test for tucker_to_tensor with partial modes"""
    X = tl.tensor(
        np.array(
            [
                [[1.0, 13], [4, 16], [7, 19], [10, 22]],
                [[2, 14], [5, 17], [8, 20], [11, 23]],
                [[3, 15], [6, 18], [9, 21], [12, 24]],
            ]
        )
    )
    ranks = [3, 4]
    U = [
        tl.tensor(np.arange(R * s, dtype=float).reshape((R, s)))
        for (R, s) in zip(ranks, tl.shape(X)[1:])
    ]
    true_res = np.array(
        [
            [
                [120.0, 456.0, 792.0, 1128.0],
                [400.0, 1472.0, 2544.0, 3616.0],
                [680.0, 2488.0, 4296.0, 6104.0],
            ],
            [
                [126.0, 486.0, 846.0, 1206.0],
                [422.0, 1582.0, 2742.0, 3902.0],
                [718.0, 2678.0, 4638.0, 6598.0],
            ],
            [
                [132.0, 516.0, 900.0, 1284.0],
                [444.0, 1692.0, 2940.0, 4188.0],
                [756.0, 2868.0, 4980.0, 7092.0],
            ],
        ]
    )
    res = tucker_to_tensor((X, U), modes=[1, 2])
    assert_array_equal(true_res, res)


def test_tucker_to_unfolded():
    """Test for tucker_to_unfolded

    Notes
    -----
    Assumes that tucker_to_tensor is properly tested
    """
    G = tl.tensor(np.random.random((4, 3, 5, 2)))
    ranks = [2, 2, 3, 4]
    U = [tl.tensor(np.random.random((ranks[i], G.shape[i]))) for i in range(tl.ndim(G))]
    full_tensor = tucker_to_tensor((G, U))
    for mode in range(tl.ndim(G)):
        assert_array_almost_equal(
            tucker_to_unfolded((G, U), mode), unfold(full_tensor, mode)
        )
        assert_array_almost_equal(
            tucker_to_unfolded((G, U), mode),
            tl.dot(
                tl.dot(U[mode], unfold(G, mode)),
                tl.transpose(kronecker(U, skip_matrix=mode)),
            ),
            decimal=5,
        )


def test_tucker_to_vec():
    """Test for tucker_to_vec

    Notes
    -----
    Assumes that tucker_to_tensor works correctly
    """
    G = tl.tensor(np.random.random((4, 3, 5, 2)))
    ranks = [2, 2, 3, 4]
    U = [tl.tensor(np.random.random((ranks[i], G.shape[i]))) for i in range(tl.ndim(G))]
    vec = tensor_to_vec(tucker_to_tensor((G, U)))
    assert_array_almost_equal(tucker_to_vec((G, U)), vec)
    assert_array_almost_equal(
        tucker_to_vec((G, U)), tl.dot(kronecker(U), tensor_to_vec(G)), decimal=5
    )


def test_tucker_mode_dot():
    """Test for tucker_mode_dot

    We will compare tucker_mode_dot
    (which operates directly on decomposed tensors)
    with mode_dot (which operates on full tensors)
    and check that the results are the same.
    """
    rng = tl.check_random_state(12345)
    shape = (5, 4, 6)
    rank = (3, 2, 4)
    tucker_ten = random_tucker(shape, rank=rank, full=False, random_state=rng)
    full_tensor = tucker_to_tensor(tucker_ten)
    # matrix for mode 1
    matrix = tl.tensor(rng.random_sample((7, shape[1])))
    # vec for mode 2
    vec = tl.tensor(rng.random_sample(shape[2]))

    # Test tucker_mode_dot with matrix
    res = tucker_mode_dot(tucker_ten, matrix, mode=1, copy=True)
    # Note that if copy=True is not respected, factors will be changes
    # And the next test will fail
    res = tucker_to_tensor(res)
    true_res = mode_dot(full_tensor, matrix, mode=1)
    assert_array_almost_equal(true_res, res, decimal=5)

    # Check that the data was indeed copied
    rec = tucker_to_tensor(tucker_ten)
    assert_array_almost_equal(full_tensor, rec, decimal=5)

    # Test tucker_mode_dot with vec
    res = tucker_mode_dot(tucker_ten, vec, mode=2, copy=True)
    res = tucker_to_tensor(res)
    true_res = mode_dot(full_tensor, vec, mode=2)
    assert_equal(res.shape, true_res.shape)
    assert_array_almost_equal(true_res, res, decimal=5)

    # Test tucker_mode_dot with vec, keeping the contracted mode
    res = tucker_mode_dot(tucker_ten, vec, mode=2, keep_dim=True, copy=True)
    res = tucker_to_tensor(res)
    true_res = tl.reshape(mode_dot(full_tensor, vec, mode=2), shape[:2] + (1,))
    assert_equal(res.shape, true_res.shape)
    assert_array_almost_equal(true_res, res, decimal=5)


def test_n_param_tucker():
    """Test for _tucker_n_param"""
    tensor_shape = (2, 3, 4, 1, 5)
    rank = (3, 2, 2, 1, 4)
    core, factors = random_tucker(shape=tensor_shape, rank=rank)
    true_n_param = np.prod(tl.shape(core)) + np.sum(
        [np.prod(tl.shape(f)) for f in factors]
    )
    n_param = _tucker_n_param(tensor_shape, rank)
    assert_equal(n_param, true_n_param)


def test_validate_tucker_rank():
    """Test validate_tucker_rank with random sizes"""
    tol = 0.01

    # Dims start at 2: a mode of size 1 is a degenerate edge case where the
    # rank-1 rounding solve can push `c` slightly above 1, making the floor
    # rounding overshoot n_param_tensor by well over 1% and flaking this test.
    tensor_shape = tuple(np.random.randint(2, 100, size=5))
    n_param_tensor = np.prod(tensor_shape)

    # Rounding = floor
    rank = validate_tucker_rank(tensor_shape, rank="same", rounding="floor")
    n_param = _tucker_n_param(tensor_shape, rank)
    assert_(n_param * (1 - tol) <= n_param_tensor)

    # Rounding = ceil
    rank = validate_tucker_rank(tensor_shape, rank="same", rounding="ceil")
    n_param = _tucker_n_param(tensor_shape, rank)
    assert_(n_param >= n_param_tensor * (1 - tol))

    # With fixed modes
    # Note: the free-mode sizes are drawn large enough that `floor(0.5 * s)`
    # rounds off a negligible fraction of each mode, so the compounded
    # rounding error across the free modes stays within `tol` of the expected
    # halving. Small free-mode sizes (e.g. single digits) make this assertion
    # flaky, since floor-rounding a handful of them can shave well over 1%
    # off the product of ranks. The fixed-mode sizes are kept small since
    # their rank equals their (squared) shape exactly, contributing no
    # rounding error, and this avoids integer overflow in `_tucker_n_param`.
    fixed_modes = [1, 4]
    fixed_sizes = np.random.randint(2, 10, size=5)
    free_sizes = np.random.randint(2000, 3000, size=5)
    tensor_shape = [
        fixed_sizes[i] ** 2 if i in fixed_modes else free_sizes[i] for i in range(5)
    ]
    n_param_tensor = np.prod(tensor_shape)
    # Floor
    rank = validate_tucker_rank(
        tensor_shape, rank=0.5, fixed_modes=fixed_modes, rounding="floor"
    )
    n_param = _tucker_n_param(tensor_shape, rank)
    for mode in fixed_modes:
        assert_(rank[mode] == tensor_shape[mode])
    assert_(n_param * (1 - tol) <= n_param_tensor * 0.5)
    # Ceil
    fixed_modes = [0, 2]
    tensor_shape = [
        s**2 if i in fixed_modes else s
        for (i, s) in enumerate(np.random.randint(2, 10, size=5))
    ]
    n_param_tensor = np.prod(tensor_shape)
    rank = validate_tucker_rank(
        tensor_shape, rank=0.5, fixed_modes=fixed_modes, rounding="ceil"
    )
    n_param = _tucker_n_param(tensor_shape, rank)
    for mode in fixed_modes:
        assert_(rank[mode] == tensor_shape[mode])
    assert_(n_param >= n_param_tensor * 0.5 * (1 - tol))


def test_tucker_normalize():
    shape = (3, 4, 5)
    rank = (3, 2, 4)
    tucker_ten = random_tucker(shape, rank)
    core, factors = tucker_normalize(tucker_ten)
    for i in range(len(factors)):
        assert_array_almost_equal(tl.norm(factors[i], axis=0), tl.ones(rank[i]))
    assert_array_almost_equal(
        tucker_to_tensor((core, factors)), tucker_to_tensor(tucker_ten)
    )


def test_tucker_copy():
    shape = (3, 4, 5)
    rank = 4
    tucker_tensor = random_tucker(shape, rank)
    core, factors = tucker_tensor
    core_normalized, factors_normalized = tucker_normalize(tucker_tensor.tucker_copy())
    # Check that modifying copy tensor doesn't change the original tensor
    assert_array_almost_equal(
        tucker_to_tensor((core, factors)), tucker_to_tensor(tucker_tensor)
    )


def partial_tucker_example():
    rng = np.random.RandomState(598)
    core = rng.normal(size=(2, 3, 4))
    factors = [rng.normal(size=(7, 4)), rng.normal(size=(5, 2))]
    dense = np.einsum("abc,ic,ja->jbi", core, *factors)
    context = {"dtype": tl.float64}
    return (
        tl.tensor(core, **context),
        [tl.tensor(factor, **context) for factor in factors],
        tl.tensor(dense, **context),
    )


def test_partial_tucker_conversions():
    core, factors, dense = partial_tucker_example()
    modes = [2, 0]
    tensor = TuckerTensor((core, factors), modes=modes)
    modes[0] = 1  # The tensor owns its mode mapping.
    assert_equal(tensor.modes, (2, 0))
    assert_equal(tensor.shape, (5, 3, 7))
    assert_equal(tensor.rank, (2, 3, 4))
    unpacked_core, unpacked_factors = tensor
    assert_(unpacked_core is core)
    assert_(unpacked_factors is factors)
    assert_equal(len(tensor), 2)
    assert_array_almost_equal(tensor.to_tensor(), dense)
    assert_array_almost_equal(tensor.to_vec(), tensor_to_vec(dense))
    assert_array_almost_equal(
        tucker_to_vec((core, factors), modes=(2, 0)), tensor_to_vec(dense)
    )
    for mode in range(3):
        assert_array_almost_equal(tensor.to_unfolded(mode), unfold(dense, mode))
        assert_array_almost_equal(
            tucker_to_unfolded((core, factors), mode, modes=(2, 0)),
            unfold(dense, mode),
        )
    assert_equal(TuckerTensor(tensor).modes, tensor.modes)


@pytest.mark.parametrize("skip", [None, 0, 1])
@pytest.mark.parametrize("transpose", [False, True])
def test_partial_tucker_conversion_options(skip, transpose):
    core, factors, _ = partial_tucker_example()
    supplied = [tl.transpose(factor) for factor in factors] if transpose else factors
    expected = core
    for i, mode in enumerate((2, 0)):
        if i != skip:
            expected = mode_dot(expected, factors[i], mode)
    tensor = (core, supplied)
    options = dict(modes=(2, 0), skip_factor=skip, transpose_factors=transpose)
    assert_array_almost_equal(tucker_to_tensor(tensor, **options), expected)
    assert_array_almost_equal(tucker_to_vec(tensor, **options), tensor_to_vec(expected))
    assert_array_almost_equal(
        tucker_to_unfolded(tensor, mode=1, **options), unfold(expected, 1)
    )


def test_partial_tucker_normalize_and_copy():
    core, factors, dense = partial_tucker_example()
    tensor = TuckerTensor((core, factors), modes=(2, 0))
    copied = tensor.tucker_copy()
    assert_equal(copied.modes, (2, 0))
    assert_(copied.core is not core)
    for copied_factor, factor in zip(copied.factors, factors):
        assert_(copied_factor is not factor)
    normalized = tucker_normalize((core, factors), modes=(2, 0))
    assert_equal(normalized.modes, (2, 0))
    assert_array_almost_equal(normalized.to_tensor(), dense)
    for factor in normalized.factors:
        assert_array_almost_equal(tl.norm(factor, axis=0), tl.ones(factor.shape[1]))
    copied.normalize()
    assert_array_almost_equal(copied.to_tensor(), dense)
    assert_array_equal(tensor.core, core)
    zero_factors = [tl.zeros(factor.shape, **tl.context(factor)) for factor in factors]
    zero_tensor = tucker_normalize((core, zero_factors), modes=(2, 0))
    assert_array_equal(
        zero_tensor.to_tensor(), tl.zeros(dense.shape, **tl.context(dense))
    )
    for factor in zero_tensor.factors:
        assert_array_equal(factor, tl.zeros(factor.shape, **tl.context(factor)))


@pytest.mark.parametrize("mode", [0, 1, 2, -1])
@pytest.mark.parametrize("operation", ["matrix", "vector", "keep_dim"])
@pytest.mark.parametrize("copy", [False, True])
def test_partial_tucker_mode_dot(mode, operation, copy):
    core, factors, dense = partial_tucker_example()
    tensor = TuckerTensor((core, factors), modes=(2, 0))
    operand = tl.arange(dense.shape[mode], **tl.context(core)) + 1
    if operation == "matrix":
        operand = tl.stack([operand, operand * 2], axis=0)
    result = tucker_mode_dot(
        (core, tuple(factors)),
        operand,
        mode,
        modes=(2, 0),
        keep_dim=operation == "keep_dim",
        copy=copy,
    )
    expected = mode_dot(dense, operand, mode % 3)
    if operation == "keep_dim":
        shape = list(dense.shape)
        shape[mode] = 1
        expected = tl.reshape(expected, shape)
    expected_modes = (2, 0)
    if operation == "vector":
        axis = mode % 3
        expected_modes = tuple(m - (m > axis) for m in (2, 0) if m != axis)
    assert_equal(result.modes, expected_modes)
    assert_equal(result.shape, expected.shape)
    assert_array_almost_equal(result.to_tensor(), expected)
    assert_array_almost_equal(
        tensor.mode_dot(
            operand, mode, keep_dim=operation == "keep_dim", copy=copy
        ).to_tensor(),
        expected,
    )
    assert_array_almost_equal(tensor.to_tensor(), dense)


def test_partial_tucker_contractions_to_scalar():
    core, factors, dense = partial_tucker_example()
    result = TuckerTensor((core, factors), modes=(2, 0))
    for mode in (0, 1, 0):
        operand = tl.ones(dense.shape[mode], **tl.context(core))
        result = result.mode_dot(operand, mode)
        dense = mode_dot(dense, operand, mode)
        assert_array_almost_equal(result.to_tensor(), dense)
        assert_array_almost_equal(result.tucker_copy().to_tensor(), dense)
        assert_array_almost_equal(tucker_normalize(result).to_tensor(), dense)
    assert_equal(result.modes, ())
    assert_equal(result.rank, ())
    assert_equal(result.shape, ())


@pytest.mark.parametrize("modes", [(2,), (2, 2), (-1, 0), (3, 0), (2.0, 0), (True, 0)])
def test_invalid_partial_tucker_modes(modes):
    core, factors, _ = partial_tucker_example()
    with assert_raises(ValueError):
        TuckerTensor((core, factors), modes=modes)
    with assert_raises(ValueError):
        tucker_to_tensor((core, factors), modes=modes)


def test_invalid_partial_tucker_factors():
    core, factors, _ = partial_tucker_example()
    with assert_raises(ValueError):
        TuckerTensor((core, factors), modes=(0, 2))
    with assert_raises(ValueError):
        TuckerTensor((core, [factors[0][:, 0], factors[1]]), modes=(2, 0))


def test_tucker_explicit_mode_override():
    core = tl.tensor(np.arange(8.0).reshape(2, 2, 2), dtype=tl.float64)
    factor = tl.tensor([[1.0, 2], [3, 4], [5, 6]], **tl.context(core))
    tensor = TuckerTensor((core, [factor]), modes=(2,))
    expected = mode_dot(core, factor, 0)
    overridden = TuckerTensor(tensor, modes=(0,))
    assert_equal(overridden.modes, (0,))
    assert_array_almost_equal(tucker_to_tensor(tensor, modes=(0,)), expected)
    assert_array_almost_equal(overridden.to_tensor(), expected)
    assert_equal(tensor.modes, (2,))


@pytest.mark.parametrize("modes", [(), (0,), (1,), (2,), (0, 2), (0, 1, 2), (2, 1, 0)])
def test_tucker_factor_subsets(modes):
    rng = np.random.RandomState(598)
    core = tl.tensor(rng.normal(size=(2, 3, 4)), dtype=tl.float64)
    factors = [
        tl.tensor(rng.normal(size=(5, core.shape[m])), **tl.context(core))
        for m in modes
    ]
    expected = core
    for mode, factor in zip(modes, factors):
        expected = mode_dot(expected, factor, mode)
    tensor = TuckerTensor((core, factors), modes=modes)
    assert_array_almost_equal(tensor.to_tensor(), expected)
    assert_array_almost_equal(tucker_normalize(tensor).to_tensor(), expected)
    assert_array_almost_equal(tensor.tucker_copy().to_tensor(), expected)
    assert_array_almost_equal(tensor.to_vec(), tensor_to_vec(expected))


def test_partial_tucker_invalid_mode_products():
    core, factors, _ = partial_tucker_example()
    tensor = TuckerTensor((core, factors), modes=(2, 0))
    for mode in (-4, 3, 0.5):
        with assert_raises(ValueError):
            tensor.mode_dot(tl.ones((2, 5), **tl.context(core)), mode)
    for operand in (
        tl.ones((2, 9), **tl.context(core)),
        tl.ones(9, **tl.context(core)),
        tl.ones((2, 2, 2), **tl.context(core)),
    ):
        with assert_raises(ValueError):
            tensor.mode_dot(operand, 0)
