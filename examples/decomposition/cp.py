"""
Canonical Polyadic Decomposition
================================
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
from tensorly.decomposition import parafac

# N, size of matrix. R, rank of data
N = 30
R = 3

# make fake data
factors = list(np.random.randn(3,N,R).astype(np.float32))
data = np.zeros((N,N,N))

# make data
for i,j,k,r in itr.product(range(N), range(N), range(N), range(R)):
    data[i,j,k] += factors[0][i,r]*factors[1][j,r]*factors[2][k,r]

# fit CP decomposition
model = parafac(data, R, verbose=1)
