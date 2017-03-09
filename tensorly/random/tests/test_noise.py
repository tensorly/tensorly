from ..noise import add_noise
from ..noise import patch_noise

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


def test_gen_noise():
    """Test for gen_noise"""

    ## Test Gaussian noise
    tol = 10e-3
    shape = (40, 3, 5, 1)
    mean = 2.5
    std = 0.0001
    tensor = np.random.random((shape))
    res = add_noise(tensor=tensor, noise='gaussian',
                    mean=mean, std=std, inplace=False)
    assert (tensor is not res)
    res -= tensor
    assert(res.shape == shape)
    assert(abs(np.mean(res) - mean) <= tol)
    assert(abs(np.std(res) - std) <= tol)

    # Inplace test
    cpy_tensor = np.copy(tensor)
    res2 = add_noise(tensor=tensor, noise='gaussian', mean=mean, std=std,
                     inplace=True)
    res2 -= cpy_tensor
    assert(res2 is tensor)
    assert(res.shape == shape)
    assert(abs(np.mean(res) - mean) <= tol)
    assert(abs(np.std(res) - std) <= tol)

    ## Test salt pepper
    tol = 10e-3
    shape = (40, 2, 4)
    percent = 0.5
    pepper_value = 0
    salt_value = 255
    # Only the noise should be exactly equal to 255 for test
    tensor = np.round(np.random.random((shape))*254)

    # Test for 0 percent of noise
    res = add_noise(tensor=tensor, noise='salt_pepper', percent=0,
                    inplace=False)
    assert_array_equal(tensor, res)

    # 50% of noise (also percent=50 should be same as 0.5)
    pepper_value = 1
    salt_value = -1
    tensor = np.random.random(shape)
    res = add_noise(tensor=tensor, noise='salt_pepper', percent=50,
                    inplace=False)
    assert (tensor is not res)
    assert (res.shape == shape)
    size = res.size
    n_noise = round(tensor.size * percent)
    n_salt = n_noise // 2
    n_pepper = n_noise - n_salt
    assert (np.sum(res == pepper_value) == n_pepper)
    assert (np.sum(res == salt_value) == n_salt)
    assert (np.sum(res == tensor) == (size - n_noise))

    # Inplace test
    cpy_tensor = np.copy(tensor)
    res2 = add_noise(tensor=tensor, noise='salt_pepper', percent=0.5,
                     inplace=True)
    assert (res2 is tensor)
    assert (np.sum(res == pepper_value) == n_pepper)
    assert (np.sum(res == salt_value) == n_salt)
    assert (np.sum(res != cpy_tensor) == n_noise)

    # Equivalence of percent as float or int:
    res1 = add_noise(tensor=tensor, noise='salt_pepper', percent=0.2,
                     inplace=True, random_state=1234)
    res2 = add_noise(tensor=tensor, noise='salt_pepper', percent=20,
                     inplace=True, random_state=1234)
    assert_array_equal(res1, res2)

    # Error case
    with assert_raises(ValueError):
        # No such thing as 200% noise
        res2 = add_noise(tensor=tensor, noise='salt_pepper', percent=200,
                         inplace=True)

    ## Test for edge cases
    with assert_raises(ValueError):
        res = add_noise(tensor=tensor, noise='does_not_exist',
                        percent=0.5, inplace=False)

def test_patch_noise():
    """Test for patch_noise"""
    tensor = np.random.random((10, 30, 40, 10))
    copy_tensor = np.copy(tensor)
    patch_size = (10, 15, 5)
    n_elements_patch = np.prod(patch_size)
    res = patch_noise(tensor, patch_size=patch_size, noise='gaussian',
                      mean=2, std=1, inplace=False)
    assert_array_equal(tensor, copy_tensor)
    for i in range(tensor.shape[0]):
        assert(np.sum(res[i] != copy_tensor[i]) == n_elements_patch)

    res = patch_noise(tensor, patch_size=patch_size, noise='gaussian',
                      mean=2, std=1, inplace=True)
    assert(tensor is res)
    for i in range(tensor.shape[0]):
        assert (np.sum(res[i] != copy_tensor[i]) == n_elements_patch)

    tensor = np.random.random((10, 30, 40, 10))
    copy_tensor = np.copy(tensor)
    noise_value = -10000
    res = patch_noise(tensor, patch_size=patch_size, noise=noise_value,
                      inplace=True)
    assert(tensor is res)
    for i in range(tensor.shape[0]):
        assert (np.sum(res[i] == noise_value) == n_elements_patch)
