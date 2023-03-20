"""
The :mod:`tensorly.datasets` module includes utilities to load datasets and
create synthetic data, e.g. for testing purposes.
"""

from .synthetic import gen_image
from .data_imports import (
    load_IL2data,
    load_covid19_serology,
    load_indian_pines,
    load_kinetic,
)
