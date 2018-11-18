# root ipython in tensorly folder
# cd Online/tensorly/ 

import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition import tucker
from math import ceil

# Copied from the online doc, to be improved

import tensorly.contrib.optimization.optim_candecomp_parafac as cpdfuns

random_state = 12345

image = tl.tensor(imresize(face(), 0.3), dtype='float64')

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# Rank of the CP decomposition
cp_rank = 25
# Rank of the Tucker decomposition
tucker_rank = [100, 100, 2]

# Perform the CP decomposition
factors, errors = cpdfuns.parafac(image, rank=cp_rank, init='random', tol=10e-6,
        return_errors=True, n_iter_max=1000 )
factors_tl, errors_tl = parafac(image, rank=cp_rank, init='random', tol=10e-6,
        return_errors=True, n_iter_max=1000)

# Perform Nonnegative CP decomposition
factors_nn, errors_nn = cpdfuns.parafac(image, rank=cp_rank, init='random', tol=10e-6,
        return_errors=True, method='NALS', n_iter_max=1000)
factors_nn_tl = non_negative_parafac(image, rank=cp_rank, init='random',
        tol=10e-6, n_iter_max=1000)

# Reconstruct the image from the factors
cp_reconstruction = tl.kruskal_to_tensor(factors)
cp_reconstruction_tl = tl.kruskal_to_tensor(factors_tl)
cp_reconstruction_nn = tl.kruskal_to_tensor(factors_nn)
cp_reconstruction_nn_tl = tl.kruskal_to_tensor(factors_nn_tl)

# plotting original and reconstructions
# Plotting the original and reconstruction from the decompositions
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
ax.set_axis_off()
ax.imshow(to_image(image))
ax.set_title('original')

ax = fig.add_subplot(2, 3, 2)
ax.set_axis_off()
ax.imshow(to_image(cp_reconstruction))
ax.set_title('CP optim')

ax = fig.add_subplot(2, 3, 3)
ax.set_axis_off()
ax.imshow(to_image(cp_reconstruction_tl))
ax.set_title('CP')

ax = fig.add_subplot(2, 3, 4)
ax.set_axis_off()
ax.imshow(to_image(cp_reconstruction_nn))
ax.set_title('CP optim NN')

ax = fig.add_subplot(2, 3, 5)
ax.set_axis_off()
ax.imshow(to_image(cp_reconstruction_nn_tl))
ax.set_title('CP NN')

plt.tight_layout()
plt.show()

