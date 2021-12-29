
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing of applying TTOI

@author: Lili Zheng
"""

import numpy as np
import math
import tensorly as tl
from tensorly import random
import tensorly.decomposition
from tensorly.contrib.decomposition.tt_TTOI import TTOI 

def test_TTOI(monkeypatch):
    """Test for the tt_TTOI function (Tensor train orthogonal iteration)
    """
    rng = tl.check_random_state(1234)
    rank = (1, 1, 1, 1, 1, 1)
    shape = (20, 20, 20, 20, 20)
    niter = 2
    
    # Generate tensor X with low tensor train rank, and its noisy observation Y
    X = random.random_tt(shape=shape, rank=rank, random_state=rng, full = True)
    context = tl.context(X)
    E = tl.tensor(np.random.normal(0,2,size = shape))
    Y = tl.tensor(X + E,**context)
    
    # run TTOI
    factors_list, full_tensor_list = TTOI(Y = Y, rank = rank, niter = niter, trajectory = True, **context)
    
    # Check that the approximation error monotonically decreases
    approx_errors = np.asarray([tl.norm(full_tensor_list[i]-Y,2) for i in range(niter*2)])
    approx_errors /= tl.norm(Y, 2)
    assert_(np.all(np.diff(approx_errors) <= 1e-3))
    
    # Check that the estimation error monotonically decreases
    estimation_errors = np.asarray([tl.norm(full_tensor_list[i]-X,2) for i in range(niter*2)])
    estimation_errors /= tl.norm(X, 2)
    assert_(np.all(np.diff(estimation_errors) <= 1e-3))
    
    # check total improvement of estimation error of TTOI from initialization (TTSVD) is larger than 10% of the norm of the true tensor
    assert_(np.all(estimation_errors[0]-estimation_errors[2*niter-1] >= 0.1))

    assert_class_wrapper_correctly_passes_arguments(monkeypatch, TTOI, TensorTrain, ignore_args={}, rank=3)
    
