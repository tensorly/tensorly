# -*- coding: utf-8 -*-
"""
Basic tensor operations
=======================

Example on how to use :mod:`tensorly` to perform basic tensor operations.

"""
import numpy as np
import tensorly as tl
from tensorly.testing import assert_array_equal

###########################################################################
# A tensor is simply a numpy array
tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)))
print('* original tensor:\n{}'.format(tensor))

###########################################################################
# Unfolding a tensor is easy
for mode in range(tensor.ndim):
    print('* mode-{} unfolding:\n{}'.format(mode, tl.unfold(tensor, mode)))

###########################################################################
# Re-folding the tensor is as easy:
for mode in range(tensor.ndim):
    unfolding = tl.unfold(tensor, mode)
    folded = tl.fold(unfolding, mode, tensor.shape)
    assert_array_equal(folded, tensor)
