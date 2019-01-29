# root ipython in tensorly folder
# cd ~/.../tensorly/

import tensorly as tl
import numpy as np
import tensorly.contrib.optimization.optim_routines as mytl

#---------- Testing nnlsHALS --------------#

# Generate problem matrices
n = 10
m = 20
r = 5
U = np.random.rand(n,r)
V = np.random.rand(r,m)
M = tl.dot(U,V)

# Computing inner products
UtU = tl.dot(tl.transpose(U),U)
UtM = tl.dot(tl.transpose(U),M)

# Testing the function
Ve = np.random.rand(r,m)
err = mytl.nnlsHALS(UtM, UtU, Ve, maxiter = 500)[1]

# Comparing the results
print(tl.norm(V-Ve)/tl.norm(V))
