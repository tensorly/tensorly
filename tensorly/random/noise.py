import numpy as np
from random import randint
from .base import check_random_state

# Author: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>


def add_noise(tensor, noise='gaussian', inplace=False, random_state=None,
              mean=0, std=1,  # Gaussian noise
              percent=0.2, salt_value=None, pepper_value=None  # salt_pepper
              ):
    """Generates noise with the specified parameters over the whole tensor

        (independent noise for each pixel, sampled from the same distribution)

    Parameters
    ----------
    tensor : ndarray on which to add noise
    noise : {'gaussian', 'salt_pepper'}
    mean : float
        mean of the noise of Gaussian noise
    std : float
        standard deviation of the Gaussian noise
    percent : float between 0 and 1
        percentage of noise to add if noise is 'salt_pepper'
    inplace : bool, optional
        if True, the noise is added inplace to tensor
    random_state : {None, int, np.random.RandomState}
        if not None, used to set the seed
    salt_value, pepper_value : float, optional
        if not specified, these are set to:

         * -1 and 1 if the arrays values are between -1 and 1
         * 0 and 255 otherwise


    Returns
    -------
    ndarray
        tensor with noise
    """
    correct_types = ['gaussian', 'salt_pepper']
    rns = check_random_state(random_state)

    if noise is 'gaussian':
        if inplace:
            tensor += rns.normal(loc=mean, scale=std, size=tensor.shape)
        else:
            tensor = tensor + rns.normal(loc=mean, scale=std, size=tensor.shape)
        return tensor

    elif noise is 'salt_pepper':
        if percent > 1 and percent < 100:
            percent /= 100
        elif percent == 1 or percent < 0 or percent > 100:
            raise ValueError('For salt and pepper noise, you need 1 > percent > 0,'
                             'but given percent={}'.format(percent))
        p = percent / 2

        if pepper_value is None:
            if np.abs(tensor).max() > 1:
                pepper_value = 255
            else:
                pepper_value = 1

        if salt_value is None:
            if pepper_value == 255:
                salt_value = 0
            else:
                salt_value = -pepper_value

        original_shape = tensor.shape
        if inplace:
            tensor.resize(tensor.size)
        else:
            tensor = np.copy(tensor).ravel()

        n_noise = round(tensor.size * percent)
        n_salt = n_noise // 2

        indices = np.arange(tensor.size)
        rns.shuffle(indices)
        tensor[indices[:n_salt]] = salt_value
        tensor[indices[n_salt:n_noise]] = pepper_value
        tensor.resize(original_shape)

        return tensor

    else:
        raise ValueError('{} is not a correct type of noise,'
                         'should be in {}'.format(noise, correct_types))


def patch_noise(tensor, patch_size, noise='gaussian',
                random_state=None, inplace=True,
                mean=0, std=1, percent=0.2):
    """Adds patches of noise on each sample of the tensor

    Parameters
    ----------
    tensor : ndarray
        tensor of shape (n_samples, n_1, ..., n_s)
        the first dimension corresponds to the samples
    patch_size : tuple
    type : {'gaussian', 'salt_pepper'} or float
        if int, the values inside the patches will simply be set to that value
    mean : float, default is 0
    std : float, default is 1
    percent: float, default is 0.2
    inplace : bool, optional
        if True the noise is added inplace
        otherwise a copy is created

    Returns
    -------
    tensor with noise
    """
    n_samples = tensor.shape[0]
    sample_shape = list(tensor.shape[1:])
    patch_size = list(patch_size)

    if not inplace:
        tensor = np.copy(tensor)

    for i in range(n_samples):
        # Generate a random patch position
        patch_position = [i]
        for j, size in enumerate(patch_size):
            start = randint(0, sample_shape[j] - size)
            patch_position.append(slice(start, start + size))
        
        if isinstance(noise, float) or isinstance(noise, int):
            tensor[patch_position] = noise
        else:
            tensor[patch_position] = add_noise(tensor=tensor[patch_position],
                                               noise=noise, mean=mean, std=std,
                                               percent=percent,
                                               random_state=random_state,
                                               inplace=inplace)

    return tensor
