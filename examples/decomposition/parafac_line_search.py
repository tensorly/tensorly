"""
Using line search with PARAFAC
==========================================

Example on how to use :func:`tensorly.decomposition.parafac` with line search to accelerate convergence.
"""

from time import time
import numpy as np
import tensorly as tl
from tensorly.random import random_kruskal
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt

tol = np.logspace(-1, -9)
err = np.empty_like(tol)
err_ls = np.empty_like(tol)
tt = np.empty_like(tol)
tt_ls = np.empty_like(tol)
tensor = random_kruskal((10, 10, 10), 3, random_state=1234, full=True)

# Get a high-accuracy decomposition for comparison
fac = parafac(tensor, rank=3, n_iter_max=2000000, tol=1.0e-15, linesearch=True)
err_min = tl.norm(tl.kruskal_to_tensor(fac) - tensor)

for ii, toll in enumerate(tol):
	# Run PARAFAC decomposition without line search and time
    start = time()
    fac = parafac(tensor, rank=3, n_iter_max=2000000, tol=toll)
    tt[ii] = time() - start
    # Run PARAFAC decomposition with line search and time
    start = time()
    fac_ls = parafac(tensor, rank=3, n_iter_max=2000000, tol=toll, linesearch=True)
    tt_ls[ii] = time() - start

    # Calculate the error of both decompositions
    err[ii] = tl.norm(tl.kruskal_to_tensor(fac) - tensor)
    err_ls[ii] = tl.norm(tl.kruskal_to_tensor(fac_ls) - tensor)

plt.loglog(tt, err - err_min, '.', label="No line search")
plt.loglog(tt_ls, err_ls - err_min, '.r', label="Line search")
plt.ylabel("Time")
plt.xlabel("Error")
plt.legend()
plt.show()