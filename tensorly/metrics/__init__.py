"""
The :mod:`tensorly.metrics` module includes utilities to measure performance
(e.g. regression error).
"""

from .regression import RMSE, MSE
from .entropy import vonneumann_entropy, tt_vonneumann_entropy, tt_mps_entanglement_entropy, cp_vonneumann_entropy
