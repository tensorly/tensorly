#!/usr/bin/env python

import time
import tensorly as tl
import numpy as np
import tensorly.contrib.optimization.optim_parafac as paraclass
import matplotlib.pyplot as plt

#--------------- Testint nnlsHALS ------------#

#----------- test 1 : small dim vs rank --------#
#
## Generate problem factors and data tensor
#n = [250,2,20000]
#r = 10
#sig = 0.001
#A = np.random.rand(n[0],r)
#B = np.random.rand(n[1],r)
#C = np.random.rand(n[2],r)
#T = tl.kruskal_to_tensor([A,B,C])
#
#T = T + sig*tl.reshape(np.random.normal(0,1,np.prod(n)),n)
#
## Defining the model
#choice = 'acc' # accelerating or not the HALS
#choice2= None
#tol = 1e-5
#alpha = 1
#delta = 0.01
#
#model = paraclass.Parafac(rank=r, verbose=True, init='random',
#        n_iter_max=1000,constraints=['NN','NN','NN','NN'], method='HALS',
#        halsacc=choice, tol=tol, alpha_hals=alpha, delta_hals=delta)
#
#fac, errors, tocacc = model.fit(T)
#initial_fact = model.init_factors
#
#model2= paraclass.Parafac(rank=r, verbose=True, init='random',
#        n_iter_max=1000,constraints=['NN','NN','NN','NN'], method='HALS',
#        halsacc=choice2, tol=tol, init_factors = initial_fact)
#
#fac2, errors2, toc = model2.fit(T)
#
#print(tocacc[-1],toc[-1])
#print(errors[-1],errors2[-1])
#
#plt.figure(1)
#plt.semilogy(list(range(len(errors))),errors,'b',
#             list(range(len(errors2))),errors2,'k')
#plt.figure(2)
#plt.semilogy(tocacc,errors,'b',
#             toc,errors2,'k')
#plt.show()
#
# -------test 2 : large number of modes --------#
## Generate problem factors and data tensor
#n = [15,15,15,15,15,15]
#r = 3
#sig = 0.001
#A = np.random.rand(n[0],r)
#B = np.random.rand(n[1],r)
#C = np.random.rand(n[2],r)
#D = np.random.rand(n[3],r)
#E = np.random.rand(n[4],r)
#F = np.random.rand(n[5],r)
#T = tl.kruskal_to_tensor([A,B,C,D,E,F])
#
#T = T + sig*tl.reshape(np.random.normal(0,1,np.prod(n)),n)
#
## Defining the model
#choice = 'acc' # accelerating or not the HALS
#choice2= None
#tol = 1e-5
#alpha = 1
#delta = 0.01
#
#model = paraclass.Parafac(rank=r, verbose=True, init='random',
#        n_iter_max=1000,constraints=['NN','NN','NN','NN','NN','NN'],
#        method='HALS', halsacc=choice, tol=tol, alpha_hals=alpha, delta_hals=delta)
#
#fac, errors, tocacc = model.fit(T)
#initial_fact = model.init_factors
#
#model2= paraclass.Parafac(rank=r, verbose=True, init='random',
#        n_iter_max=1000,constraints=['NN','NN','NN','NN','NN','NN'], 
#        method='HALS',halsacc=choice2, tol=tol, init_factors = initial_fact)
#
#fac2, errors2, toc = model2.fit(T)
#
#print(tocacc[-1],toc[-1])
#print(errors[-1],errors2[-1])
#
#plt.figure(1)
#plt.semilogy(list(range(len(errors))),errors,'b',
#             list(range(len(errors2))),errors2,'k')
#plt.figure(2)
#plt.semilogy(tocacc,errors,'b',
#             toc,errors2,'k')
#plt.show()

# -------test 3 : factors colinearity --------#
## Generate problem factors and data tensor
#n = [50,50,50]
#r = 5
#sig = 0.001
#A = np.random.rand(n[0],r)
#B = np.random.rand(n[1],r)
#C = np.random.rand(n[2],r)
#
## Correlating the signal
#    # A mean
#U,S,V = np.linalg.svd(A)
#Ark1  = S[0]*np.outer(U[:,0], V[0,:])
#rho = 0.99
#A = A*(1-rho) + Ark1*rho
#
#T = tl.kruskal_to_tensor([A,B,C])
#T = T + sig*tl.reshape(np.random.normal(0,1,np.prod(n)),n)
#
## Defining the model
#choice = 'acc' # accelerating or not the HALS
#choice2= None
#tol = 1e-8
#alpha = 2
#delta = 0.01
#
#model = paraclass.Parafac(rank=r, verbose=True, init='random',
#        n_iter_max=1000,constraints=['NN','NN','NN','NN','NN','NN'],
#        method='HALS', halsacc=choice, tol=tol, alpha_hals=alpha, delta_hals=delta)
#
#fac, errors, tocacc = model.fit(T)
#initial_fact = model.init_factors
#
#model2= paraclass.Parafac(rank=r, verbose=True, init='random',
#        n_iter_max=1000,constraints=['NN','NN','NN','NN','NN','NN'], 
#        method='HALS',halsacc=choice2, tol=tol, init_factors = initial_fact)
#
#fac2, errors2, toc = model2.fit(T)
#
#print(tocacc[-1],toc[-1])
#print(errors[-1],errors2[-1])
#
#plt.figure(1)
#plt.semilogy(list(range(len(errors))),errors,'b',
#             list(range(len(errors2))),errors2,'k')
#plt.figure(2)
#plt.semilogy(tocacc,errors,'b',
#             toc,errors2,'k')
#plt.show()
#
# -------test 4 : dummy --------#
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
choice = 'acc' # accelerating or not the HALS
choice2= None
tol = 1e-5
alpha = 1
delta = 0.01

model = paraclass.Parafac(rank=r, verbose=True, init='random',
        n_iter_max=1000,constraints=['NN','NN','NN','NN','NN','NN'],
        method='HALS', halsacc=choice, tol=tol, alpha_hals=alpha, delta_hals=delta)

fac, errors, tocacc = model.fit(T)
initial_fact = model.init_factors

model2= paraclass.Parafac(rank=r, verbose=True, init='random',
        n_iter_max=1000,constraints=['NN','NN','NN','NN','NN','NN'], 
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
