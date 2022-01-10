
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing of applying TTOI

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
    
    # Generate tensor true_tensor with low tensor train rank, and its noisy observation data_tensor
    true_tensor = random.random_tt(shape = shape, rank = rank, random_state = rng, full = True)
    context = tl.context(true_tensor)
    noise_tensor = tl.tensor(np.random.normal(0,2,size = shape))
    data_tensor = tl.tensor(true_tensor + noise_tensor,**context)
    
    # run TTOI
    factors_list, full_tensor_list = TTOI(data_tensor = data_tensor, rank = rank, niter = niter, trajectory = True, **context)
    
    # Check that the approximation error monotonically decreases
    approx_errors = np.asarray([tl.norm(full_tensor_list[i]-data_tensor,2) for i in range(niter*2)])
    approx_errors /= tl.norm(data_tensor, 2)
    assert_(np.all(np.diff(approx_errors) <= 1e-3))
    
    # Check that the estimation error monotonically decreases
    estimation_errors = np.asarray([tl.norm(full_tensor_list[i]-true_tensor,2) for i in range(niter*2)])
    estimation_errors /= tl.norm(true_tensor, 2)
    assert_(np.all(np.diff(estimation_errors) <= 1e-3))
    
    # check total improvement of estimation error of TTOI from initialization (TTSVD) is larger than 10% of the norm of the true tensor
    assert_(np.all(estimation_errors[0]-estimation_errors[2*niter-1] >= 0.1))

<<<<<<< HEAD
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, TTOI, TensorTrain, ignore_args = {}, rank = 3)
=======
    assert_class_wrapper_correctly_passes_arguments(monkeypatch, TTOI, TensorTrain, ignore_args={}, rank=3)
>>>>>>> 823deefb5acdfcf4268c6727d9f7b148bc7f7e7b
    