import tensorly as tl

import operator
import time
import itertools

import numpy as np
import math
import numpy.linalg as npla
import numpy.random as npr

from scipy import stats

from TensorToolbox.core import STT
from .spectral_tensor_train_decomposition import STT
import TensorToolbox.multilinalg as mla

from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND
from UQToolbox import RandomSampling as RS

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
    return X[0] ** 2 + X[1] ** 2

n = 10
d = 4
rng = 1
grid = getEquispaceGrid(d, rng, n)

# Construct a test approximation
approximation = STT(func, grid, params=0, surrogateONOFF = True, surrogate_type='LagrangeInterpolation')
# Call the approximation
x = approximation.build()

m = 10
sample = npr.rand(10,d)
approx = approximation.__call__(sample)

true = evaluate(sample,func)
print(approx)
print(true)
print(npla.norm(true-approx))

tl.assert_(tl.norm(true-approx) < 0.01)
