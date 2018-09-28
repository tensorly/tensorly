import numpy as np

from tensorly import backend as T


def assert_array_equal(a, b, *args, **kwargs):
    np.testing.assert_array_equal(T.to_numpy(a), T.to_numpy(b),
                                  *args, **kwargs)


def assert_array_almost_equal(a, b, *args, **kwargs):
    np.testing.assert_array_almost_equal(T.to_numpy(a), T.to_numpy(b),
                                         *args, **kwargs)


def assert_equal(actual, desired, *args, **kwargs):
    def _tensor_to_numpy(x):
        if T.is_tensor(x):
            x = T.to_numpy(x)
            return x[0] if x.shape == (1,) else x
        return x

    np.testing.assert_equal(_tensor_to_numpy(actual),
                            _tensor_to_numpy(desired),
                            *args, **kwargs)


assert_ = np.testing.assert_
assert_raises = np.testing.assert_raises
