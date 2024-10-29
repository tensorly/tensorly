"""
The :mod:`tensorly.metrics` module includes utilities to measure performance
(e.g. regression error).
"""

from .regression import RMSE, MSE
from .entropy import vonneumann_entropy, tt_vonneumann_entropy, cp_vonneumann_entropy
from .factors import congruence_coefficient
from .similarity import correlation_index
from .leverage_scores import leverage_score_dist
