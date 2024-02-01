"""
Non-negative Tucker decomposition
=================================
Example and comparison of Non-negative Tucker decompositions.
"""

##############################################################################
# Introduction
# -----------------------
# Since version 0.6 in Tensorly, two algorithms are available to compute non-negative
# Tucker decomposition:
#
# 1. Multiplicative updates (MU) (already in Tensorly < 0.6)
# 2. Non-negative Alternating Least Squares (ALS) using Hierarchical ALS (HALS)
#
# Non-negativity is an important constraint to impose in Tucker decompositions.
# Tucker decomposition is indeed not readily identifiable (the same fit can be obtained using infinitely many
# choices of factors and core). Nonnegative Tucker decomposition features a nonnegative core tensor
# :math:`G` and nonnegative factors (:math:`A`, :math:`B`, :math:`C`).
#
# .. math::
#     [| G; A, B, C |],
#
# Obtaining the factors and the core from the data is a difficult optimization problem. We may resort to alternating 
# optimization strategies, which have the benefit to make use of well-understood algorithms to solve nonnegative least squares sub-problems to estimate each factor and the core tensor.
# Given a data tensor :math:`T`, to update a specific factor we solve the following problem (e.g. factor :math:`A` here):
#
# .. math::
#     \min_{A \geq 0} ||T_{[1]} - A G_{[1]}(B\otimes C)^T||_F^2,
#
# Here, :math:`G_{[i]}` represents ith mode unfolding of the core, and :math:`\otimes` represents the Kronecker product. To update
# the core, we solve the following problem:
#
# .. math::
#    \min_{g \geq 0} ||t -   (A\otimes B \otimes C) g ||_F^2,
#
# where :math:`t` and :math:`g` are the vectorized data tensor :math:`T` and core :math:`G`.


import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker, non_negative_tucker_hals
from tensorly.random import random_tucker
import time
from tensorly.metrics.regression import RMSE
import matplotlib.pyplot as plt


##############################################################################
# Create synthetic tensor
# -----------------------
# There are several ways to create a tensor with non-negative entries in Tensorly.
# Here we chose to generate a random tensor from the sequence of integers from
# 1 to 1000.

# tensor generation
rng = tl.check_random_state(12)
tensor_tucker = random_tucker(shape=(20,20,20),rank=(5,5,5),random_state=rng)
tensor_tucker[1] = [tl.abs(fac) for fac in tensor_tucker[1]]  # tensor_tucker[1] is a list of factors
tensor_tucker[0] = tl.abs(tensor_tucker[0])  # tensor_tucker[0] is the core tensor
tensor = tensor_tucker.to_tensor()

##############################################################################
# Non-negative Tucker
# -----------------------
# Upon calling ``non_negative_tucker``, the MU algorithm is used.

tic = time.time()
tensor_mu, error_mu = non_negative_tucker(tensor, rank=[5, 5, 5], tol=0, n_iter_max=2500, return_errors=True)
tucker_reconstruction_mu = tl.tucker_to_tensor(tensor_mu)
time_mu = time.time()-tic

##############################################################################
# Here, we also compute the output tensor from the decomposed factors by using
# the ``tucker_to_tensor`` function. The tensor ``tucker_reconstruction_mu`` is
# therefore a low-rank non-negative approximation of the input tensor ``tensor``.

##############################################################################
# Non-negative Tucker with HALS and FISTA
# ---------------------------------------
# The second algorithm can be called with ``non_negative_tucker_hals``.
#
# It uses HALS as a nonnegative least squares solver to update the factors. To update the core however,
# the inherent structure of the nonnegative least squares problem is different, and therefore it is reasonable to use
# the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA). FISTA is a popular accelerated gradient method for
# constrained or unconstrained problems, and it is well suited here.
#
# Note that we reduce the number of iterations for the inner nonnegative least squares solvers with respect to the default configuration because the problem is simple. Indeed the data is exactly a Tucker tensor and no noise was added.

ticnew = time.time()
error_hals = []
def callback_error(_, error):
    error_hals.append(2*error)  # hals error is halved
    
    
tensor_hals_fista = non_negative_tucker_hals(tensor, rank=[5, 5, 5], callback=callback_error, inner_iter_max=20, inner_iter_max_fista=60, tol=0)
tucker_reconstruction_fista = tl.tucker_to_tensor(tensor_hals_fista)
time_fista = time.time()-ticnew

##############################################################################
# Comparison
# -----------------------
# To compare the various methods, first we may look at each algorithm
# processing time:

print('time for MU'+' ' + str("{:.2f}".format(time_mu)))
print('time for HALS+FISTA:'+' ' + str("{:.2f}".format(time_fista)))

##############################################################################
# All algorithms should run with about the same number of iterations on our
# example, so at first glance the MU algorithm is faster (i.e. has lower
# per-iteration complexity). A second way to compare methods is to compute
# the error between the output and input tensor. In Tensorly, there is a function
# to compute Root Mean Square Error (RMSE):

print('RMSE for MU:'+' ' + str(RMSE(tensor, tucker_reconstruction_mu)))
print('RMSE for HALS+FISTA:'+' ' + str(RMSE(tensor, tucker_reconstruction_fista)))

##############################################################################
# According to the RMSE results, HALS is better than the multiplicative update
# with both FISTA and active set core update options. We can better appreciate
# the difference in convergence speed on the following error per iteration plot:


def each_iteration(a,b,title):
    fig=plt.figure()
    fig.set_size_inches(10, fig.get_figheight(), forward=True)
    plt.loglog(a)
    plt.loglog(b)
    plt.title(str(title))
    plt.legend(['MU', 'HALS + FISTA'], loc='upper right')
    plt.xlabel('Iteration index')


each_iteration(error_mu, error_hals[1:], 'Relative error at each iteration')

##############################################################################
# In conclusion, on this quick test, it appears that the HALS algorithm gives
# much better results than the MU Tensorly method. Our recommendation
# is to use HALS as a default, and only resort to MU in specific cases
# (only encountered by expert users most likely).

##############################################################################
# References
# ----------
#
# Gillis, N., & Glineur, F. (2012). Accelerated multiplicative updates and
# hierarchical ALS algorithms for nonnegative matrix factorization.
# Neural computation, 24(4), 1085-1105. 