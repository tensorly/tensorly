"""
This module will contain all functions that operate on sparse tensors.

- computation of sparse tensors: this module
- storage of sparse tensors: tensorly.backends

The functions in this module should be able to take sparse tensors from any
backend.
"""


def approximate(dense_tensor):
    """
    Perform some lossy compression on a dense tensor to approximate it as a
    sparse tensor.
    """
    pass


def robust_pca(sparse_tensor):
    """ example """
    raise NotImplementedError


def parafac(sparse_tensor):
    """ example """
    raise NotImplementedError
