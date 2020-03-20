# root ipython in tensorly folder
# cd ~/.../tensoptly/

import tensorly as tl
import numpy as np
from ..nnls_routines import hals_nnls_acc

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
err = hals_nnls_acc(UtM, UtU, Ve, maxiter=500, delta=0)[1]

# Comparing the results
print(tl.norm(V-Ve)/tl.norm(V), err)
