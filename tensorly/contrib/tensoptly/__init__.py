"""
A tentative extension of tensorly that describes decomposition problems with classes. The goal is to allow various algorithms to be run on a single data set with little effort, and to easily tune these algorithm's parameters.

Moreover, it makes it easier to add/tune existing algorithms in tensorly, in particular in the presence of constraints.

Finally, tensoptly implements a few important optimization routines such as an HALS solver for nonnegative least squares.
"""

from .parafac_class import Parafac
from .nnls_routines import hals_nnls_acc
