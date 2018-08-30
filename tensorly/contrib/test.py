import tensorly as tl

import operator
import time
import itertools

import numpy as np
import math
import numpy.linalg as npla
import numpy.random as npr
# from . import STT
from spectral_tensor_train_decomposition import STT


npr.seed(1)


def getEquispaceGrid(n_dim, rng, subdivisions):
    '''
    Returns a grid of equally-spaced points in the specified number of dimensions

    n_dim       : The number of dimensions to construct the tensor grid in
    rng         : The maximum dimension coordinate (grid starts at 0)
    subdivisions: Number of subdivisions of the grid to construct
    '''
    return np.array([np.array(range(subdivisions + 1)) * rng * 1.0 / subdivisions for i in range(n_dim)])


def evaluateGrid(grid, fcn):
    '''
    Loops over a grid in specified order and computes the specified function at each
    point in the grid, returning a list of computed values.
    '''
    values = np.zeros(len(grid[0]) ** len(grid))
    idx = 0
    for permutation in itertools.product(range(len(grid[0])), repeat=len(grid)):
        pt = np.array([grid[i][permutation[i]] for i in range(len(permutation))])
        values[idx] = fcn(pt)
        idx += 1

    return values

def evaluate(sample, func):
    '''
    Given samples x1 to xm, return y1 to ym
    '''
    (m,d) = sample.shape
    sample_value = np.zeros(m)
    for i in range(m):
        sample_value[i] = func(sample[i,:])

    return sample_value


# Quadratic polynomial
def func(X, params=0):
    return np.sum(X) ** 2

n = 10
d = 3
rng = 1
grid = getEquispaceGrid(d, rng, n)

print(len(grid))

print("-----------------------------------start---------------------------------------")

# Construct a test approximation
# approximation = STT(func, grid, params=0, eps=1e-10, method="svd", surrogate_type='LagrangeInterpolation')
approximation = STT(func, grid, params=0, eps=1e-10, method="svd", surrogate_type='LinearInterpolation')

m = 10
sample = npr.rand(10,d)
approx = approximation.__call__(sample)

print("----------------------------------- end----------------------- ----------------")

true = evaluate(sample,func)
print(approx)
print(true)
print(npla.norm(true-approx))
