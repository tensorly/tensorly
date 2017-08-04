# -*- coding: utf-8 -*-
"""
Basic tensor operations
=======================

Example on how to use :mod:`tensorly.base` to perform basic tensor operations.

"""

import matplotlib.pyplot as plt
from tensorly.base import unfold, fold
import numpy as np
import tensorly.backend as T

###########################################################################
# A tensor is simply a numpy array
tensor = T.tensor(np.arange(24).reshape((3, 4, 2)))
print('* original tensor:\n{}'.format(tensor))

###########################################################################
# Unfolding a tensor is easy
for mode in range(tensor.ndim):
    print('* mode-{} unfolding:\n{}'.format(mode, unfold(tensor, mode)))

###########################################################################
# Re-folding the tensor is as easy:
for mode in range(tensor.ndim):
    unfolding = unfold(tensor, mode)
    folded = fold(unfolding, mode, tensor.shape)
    T.assert_array_equal(folded, tensor)
