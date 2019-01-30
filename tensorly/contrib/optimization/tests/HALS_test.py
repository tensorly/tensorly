#!/usr/bin/env python

import tensorly as tl
import numpy as np
import tensorly.contrib.optimization.optim_parafac as paraclass

#--------------- Testint nnlsHALS ------------#

# Generate problem factors and data tensor
n1,n2,n3 = [10,11,12]
r = 5
A = np.random.rand(n1,r)
B = np.random.rand(n2,r)
C = np.random.rand(n3,r)
T = tl.kruskal_to_tensor([A,B,C])

# Defining the model

model = paraclass.Parafac(rank=r, verbose=False, init='random',
        n_iter_max=1000,constraints=['NN','NN','NN'], method='HALS')

factors, errors = model.fit(T)
