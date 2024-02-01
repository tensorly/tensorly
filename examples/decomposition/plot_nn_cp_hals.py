"""
Nonnegative CP decomposition in Tensorly >=0.6
===============================================
Example and comparison of nonnegative CP/PARAFAC decompositions.
"""

##############################################################################
# Introduction
# -----------------------
# Since version 0.6 in Tensorly, several options are available to compute
# Nonnegative Canonical Polyadic decomposition (NCP), that is a CP decomposition
# where factors have only nonnegative entries.
#
# 1. Multiplicative updates (MU) (already in Tensorly < 0.6)
# 2. Nonnegative Alternating Least Squares (ALS) using Hierarchical ALS (HALS)
#
# Nonnegativity is an important constraint to handle for tensor decompositions.
# In many applications, the data tensor itself is nonnegative, and one could 
# expect that factors (e.g. spectra, relative concentrations, images...)
# must have only nonnegative to ensure interpretability.
#
# Note: Example updated for Tensorly >=0.8 modified API of ``non_negative_parafac_hals``

import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_parafac_hals
from tensorly.decomposition._cp import initialize_cp
from tensorly.cp_tensor import CPTensor
from tensorly.random import random_cp
import time
from copy import deepcopy

##############################################################################
# Create synthetic tensor
# -----------------------
# There are several ways to create a tensor with nonnegative entries in Tensorly.
# Here we chose to generate a random tensor using tensorly random tools.
# The random tensor we generate has NCP structure with rank 10. The random_cp
# method does not allow to produce a NCP tensor directly, so we post-process
# the factors of an unconstrained random CP to make then nonnegative,
# then form the data tensor.

# Tensor generation
rng = tl.check_random_state(12)
tensor_cp = random_cp(shape=(20, 20, 20), rank=10, random_state=rng)
tensor_cp[1] = [tl.abs(fac) for fac in tensor_cp[1]]
tensor = tensor_cp.to_tensor()

##############################################################################
# Our goal here is to produce an approximation of the tensor generated above
# which follows a low-rank NCP model. Before
# using these algorithms, we can use Tensorly to produce a good initial guess
# for our NCP. In fact, in order to compare both algorithmic options in a
# fair way, it is a good idea to use same initialized factors in decomposition
# algorithms. We make use of the ``initialize_cp`` function to initialize the
# factors of the NCP (setting the ``non_negative`` option to `True`) 
# and transform these factors (and factors weights) into
# an instance of the CPTensor class:

weights_init, factors_init = initialize_cp(tensor, non_negative=True, init='random', rank=10, random_state=rng)

cp_init = CPTensor((weights_init, factors_init))

##############################################################################
# Nonnegative Parafac
# -----------------------
# From now on, we can use the same ``cp_init`` tensor as the initial tensor when
# we use decomposition functions. Now let us first use the algorithm based on
# Multiplicative Update, which can be called as follows:

tic = time.time()
errors_mu = []
def callback_error_mu(_,error):
    errors_mu.append(error)
    

tensor_mu = non_negative_parafac(tensor, rank=10, init=deepcopy(cp_init), n_iter_max=1100, tol=0, callback=callback_error_mu)
cp_reconstruction_mu = tl.cp_to_tensor(tensor_mu)
time_mu = time.time()-tic

##############################################################################
# Note that to obtain the error at each iteration we can invoke a user-defined
# callback function. This allows for a finer user specification on the returns
# than the ``return_error'' keyword exposed in the API of previous tensorly versions.

##############################################################################
# Here, we also compute the output tensor from the decomposed factors by using
# the cp_to_tensor function. The tensor cp_reconstruction_mu is therefore a
# low-rank nonnegative approximation of the input tensor; looking at the
# first few values of both tensors shows that this is indeed
# the case but the approximation is quite coarse.

print('reconstructed tensor\n', cp_reconstruction_mu[10:12, 10:12, 10:12], '\n')
print('input data tensor\n', tensor[10:12, 10:12, 10:12])

##############################################################################
# Nonnegative Parafac with HALS
# ------------------------------
# Our second option to compute NCP is the HALS algorithm. This algorithm has
# more options, is in general more efficient at solving the decomposition
# problem and should be the preferred method. Let us verify this statement on our random example.
# Here we set the maximal number of iterations to 100, and a stopping tolerance of 0 to ensure
# all 100 iterations are run.

errors_hals = []
def callback_error(_,error):
    errors_hals.append(2*error)  # hals error is halved
    

tic = time.time()
tensor_hals = non_negative_parafac_hals(tensor, rank=10, init=deepcopy(cp_init), callback=callback_error, tol=0, n_iter_max=50)
cp_reconstruction_hals = tl.cp_to_tensor(tensor_hals)
time_hals = time.time()-tic


##############################################################################
# Again, we can look at the reconstructed tensor entries.

print('reconstructed tensor\n',cp_reconstruction_hals[10:12, 10:12, 10:12], '\n')
print('input data tensor\n', tensor[10:12, 10:12, 10:12])

##############################################################################
# Nonnegative Parafac with tuned-parameters HALS
# ------------------------------------
# From only looking at a few entries of the reconstructed tensors, we can
# already notice a gap between HALS and MU outputs.
# Additionally, HALS algorithm has a few options to tune the algorithm.
# We may for instance decrease the number of inner iterations for the 
# nonnegative least squares solver used within the ``non_negative_parafac_hals``
# method. We can also increase the tolerance for the same inner solver.
# The tuned HALS algorithm here will make slightly worse per-iteration
# improvements to the fit, but the iterations will be much faster.

tic = time.time()
errors_hals_tuned = []
def callback_error_tuned(_, error):
    errors_hals_tuned.append(2*error)  # hals error is halved
    
    
tensor_hals_tuned = non_negative_parafac_hals(tensor, rank=10, init=deepcopy(cp_init), callback=callback_error_tuned, inner_iter_max=10, inner_tol=0.3, tol=0, n_iter_max=50)
cp_reconstruction_tuned = tl.cp_to_tensor(tensor_hals_tuned)
time_tuned = time.time()-tic

##############################################################################
# Comparison
# -----------------------
# First comparison option is processing time, then reconstruction error
# for each algorithm:

print(str("{:.2f}".format(time_mu)) + ' ' + 'seconds')
print(str("{:.2f}".format(time_hals)) + ' ' + 'seconds')
print(str("{:.2f}".format(time_tuned)) + ' ' + 'seconds')

from tensorly.metrics.regression import RMSE
print(RMSE(tensor, cp_reconstruction_mu))
print(RMSE(tensor, cp_reconstruction_hals))
print(RMSE(tensor, cp_reconstruction_tuned))

##############################################################################
# According to the RMSE results, HALS with tuned parameters is more precise
# than the multiplicative update solution for a similar runtime.
# We can better appreciate the difference
# in convergence speed on the following error per iteration plot.

import matplotlib.pyplot as plt
def each_iteration(a,b,c,title):
    fig=plt.figure()
    fig.set_size_inches(10, fig.get_figheight(), forward=True)
    plt.loglog(a)
    plt.loglog(b)
    plt.loglog(c)
    plt.title(str(title))
    plt.legend(['MU', 'HALS', 'Tuned HALS'], loc='upper left')
    plt.xlabel('Iteration index')


each_iteration(errors_mu, errors_hals, errors_hals_tuned, 'Relative squared error per iteration')

##############################################################################
# In conclusion, on this quick test, it appears that the HALS algorithm gives
# much better results than the MU original Tensorly methods. Our recommendation
# is to use HALS as a default, and only resort to MU in specific cases (only
# encountered by expert users most likely).

##############################################################################
# References
# ----------
#
# Gillis, N., & Glineur, F. (2012). Accelerated multiplicative updates and
# hierarchical ALS algorithms for nonnegative matrix factorization.
# Neural computation, 24(4), 1085-1105. (Link)
# <https://direct.mit.edu/neco/article/24/4/1085/7755/Accelerated-Multiplicative-Updates-and>
