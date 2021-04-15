"""
The :mod:`tensorly.decomposition` module includes utilities for performing
tensor decomposition such as CANDECOMP-PARAFAC and Tucker.                                                                                               
"""

from ._cp import parafac, CP, RandomizedCP, randomised_parafac, sample_khatri_rao
from ._nn_cp import non_negative_parafac, non_negative_parafac_hals, CP_NN_HALS, CP_NN
from ._tucker import tucker, partial_tucker, non_negative_tucker, Tucker
from .robust_decomposition import robust_pca
from ._tt import TensorTrain, tensor_train, tensor_train_matrix
from ._parafac2 import parafac2, Parafac2
from ._symmetric_cp import symmetric_parafac_power_iteration, symmetric_power_iteration, SymmetricCP
from ._cp_power import parafac_power_iteration, power_iteration, CPPower

# Deprecated
from ._tt import matrix_product_state

