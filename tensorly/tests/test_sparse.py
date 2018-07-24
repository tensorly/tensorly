import tensorly as tl
import tensorly.decomposition as decomp
import sparse
import pytest
import tensorly.backend as T
import numpy as np


def test_context():
    x = sparse.random((10, 10, 40))
    context = T.context(x)
    assert set(context.keys()) == {'sparse', 'coords', 'shape', 'dtype'}
    assert context['shape'] == (10, 10, 40)
    assert context['sparse']

    y = x.todense()
    assert not T.context(y).get('sparse', False)
    assert set(T.context(y).keys()) == {'dtype'}


def test_get_tensor():
    shape = (10, 20, 30)
    coords = np.random.randint(10, size=30).reshape(10, 3)
    data = np.arange(10)

    y1 = sparse.COO(coords.T, data=data, shape=shape)
    y2 = T.tensor(y1, **T.context(y1))

    assert np.allclose(y1.coords, y2.coords)
    assert np.allclose(y1.data, y2.data)
    assert y1.shape == y2.shape
    assert type(y1) == type(y2)

    y3 = T.tensor(y1, sparse=False)
    assert isinstance(y3, np.ndarray)


def test_to_numpy():
    x = sparse.random((10, 20, 10))
    y = T.to_numpy(x)
    assert isinstance(y, np.ndarray)
    assert y.shape == (10, 20, 10)


def test_array_eq():
    x = sparse.random((10, 20, 20))
    y = sparse.COO.from_numpy(x.todense())
    T.assert_array_almost_equal(x, y)
    z = y + 1e-10 * sparse.random((10, 20, 20))
    T.assert_array_almost_equal(x, z)
