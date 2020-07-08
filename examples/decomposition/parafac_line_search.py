import tensorly as tl
from tensorly.random import random_kruskal
from time import time
from tensorly.decomposition import parafac
import numpy as np
import matplotlib.pyplot as plt

tol = np.logspace(-1, -10)
err = np.empty_like(tol)
err_ls = np.empty_like(tol)
tt = np.empty_like(tol)
tt_ls = np.empty_like(tol)
tensor = random_kruskal((10, 20, 20), 5, random_state=1234, full=True)

fac = parafac(tensor, rank=3, n_iter_max=2000000, tol=1.0e-14, orthogonalise=5, linesearch=True)
err_min = tl.norm(tl.kruskal_to_tensor(fac) - tensor)

for ii, toll in enumerate(tol), total=len(tol):
    start = time()
    fac = parafac(tensor, rank=3, n_iter_max=2000000, orthogonalise=5, tol=toll)
    tt[ii] = time() - start
    start = time()
    fac_ls = parafac(tensor, rank=3, n_iter_max=2000000, orthogonalise=5, tol=toll, linesearch=True)
    tt_ls[ii] = time() - start
    
    err[ii] = tl.norm(tl.kruskal_to_tensor(fac) - tensor)
    err_ls[ii] = tl.norm(tl.kruskal_to_tensor(fac_ls) - tensor)

plt.loglog(tt, err - err_min, '.', label="No line search")
plt.loglog(tt_ls, err_ls - err_min, '.r', label="Line search")
plt.ylabel("Time")
plt.xlabel("Error")
plt.legend()
plt.show()