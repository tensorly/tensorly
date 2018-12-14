# root ipython in tensorly folder
# cd ~/.../tensorly/ 

import time
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition import tucker
from math import ceil

import tensorly.contrib.optimization.optim_parafac as paraclass

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

# Perform the CP decomposition
    # creating the class instance
model = paraclass.Parafac(cp_rank)
factors, errors = model.fit(image)

# Reconstruct the image from the factors
cp_reconstruction = tl.kruskal_to_tensor(factors)


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

plt.tight_layout()
plt.show()
