#!/usr/bin/env python
# run with tensorly in root

import tensorly as tl
import numpy as np
# import tensorly.contrib.optimization.optim_parafac as paraclass
from ..optimization import optim_parafac as paraclass
import matplotlib.pyplot as plt

#--------------- Testint nnlsHALS ------------#
# Generate problem factors and data tensor
n = [50,50,50]
r = 5
sig = 0.001
A = np.random.rand(n[0],r)
B = np.random.rand(n[1],r)
C = np.random.rand(n[2],r)
T = tl.kruskal_to_tensor([A,B,C])

T = T + sig*tl.reshape(np.random.normal(0,1,np.prod(n)),n)

# Defining the model
choice = 'acc'  # accelerating or not the HALS
choice2 = None
tol = 1e-5
alpha = 1
delta = 0.01

model = paraclass.Parafac(rank=r, verbose=True, init='random',
        n_iter_max=1000,constraints=['NN','NN','NN'],
        method='HALS', halsacc=choice, tol=tol, alpha_hals=alpha, delta_hals=delta)

fac, errors, tocacc = model.fit(T)
initial_fact = model.init_factors

model2 = paraclass.Parafac(rank=r, verbose=True, init='random',
         n_iter_max=1000,constraints=['NN','NN','NN'], 
         method='HALS',halsacc=choice2, tol=tol, init_factors = initial_fact)

fac2, errors2, toc = model2.fit(T)

print(tocacc[-1],toc[-1])
print(errors[-1],errors2[-1])

plt.subplot(2,1,1)
plt.semilogy(list(range(len(errors))),errors,'b',
             list(range(len(errors2))),errors2,'k')
plt.subplot(2,1,2)
plt.semilogy(tocacc,errors,'b',
             toc,errors2,'k')
plt.show()
