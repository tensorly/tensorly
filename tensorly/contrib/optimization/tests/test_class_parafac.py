#!/usr/bin/env python
# coding: utf-8
# root ipython in tensorly folder. No need if tensorly installed with pip with optimization module support.
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize
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


# Testing the ALS algorithm and the class definition

# Perform the CP decomposition
fixed = [2]
constraints = []
model = paraclass.Parafac(rank=cp_rank, verbose=False, init='random',
                          random_state=random_state, fixed_modes=fixed,
                          n_iter_max=1000, constraints=constraints)


factors, errors = model.fit(image)

# Reconstruct the image from the factors
cp_reconstruction = model.reconstruct()

# Testing Fast Gradient

# Perform the CP decomposition
# Choosing the Fast Gradient optimization method
algo = 'FG'
# Choosing stepsize
step = 5*1e-7
# Fixed mode test
fixed = []
# Constraints test
constraints = ['NN', '', '']
# creating the class instance
model2 = paraclass.Parafac(rank=cp_rank, verbose=False, init='random',
                           random_state=random_state, fixed_modes=fixed,
                           n_iter_max=1000, constraints=constraints,
                           method=algo, step=step, tol=1e-5)

# Fitting
factor_fg, errors_fg = model2.fit(image)

# Reconstruct the image from the factors
cp_reconstruction2 = model2.reconstruct()


# Testing Multiplicative Update
# Perform the CP decomposition with MU
# Fixed mode test
fixed = []
# Constraints test
constraints = ['NN', 'NN', 'NN']
# creating the class instance
model3 = paraclass.Parafac(rank=cp_rank, verbose=False, init='random',
                           random_state=random_state, fixed_modes=fixed,
                           n_iter_max=1000, constraints=constraints,
                           tol=1e-8)

# Different syntax, e.g. user wants to correct method after warning
model3.method = 'MU'

# Computing
factor_mu, errors_mu = model3.fit(image)

# Reconstruct the image from the factors
cp_reconstruction3 = model3.reconstruct()


# # Comparative plots

# Plotting the original and reconstruction from the decompositions
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.set_axis_off()
ax.imshow(to_image(image))
ax.set_title('original')

ax = fig.add_subplot(2, 2, 2)
ax.set_axis_off()
ax.imshow(to_image(cp_reconstruction))
ax.set_title('ALS unconstrained CP,\n fixed third mode')

ax = fig.add_subplot(2, 2, 3)
ax.set_axis_off()
ax.imshow(to_image(cp_reconstruction2))
ax.set_title('FG partially constrained CP')

ax = fig.add_subplot(2, 2, 4)
ax.set_axis_off()
ax.imshow(to_image(cp_reconstruction3))
ax.set_title('MU nonnegative CP')

plt.tight_layout()
plt.show()
