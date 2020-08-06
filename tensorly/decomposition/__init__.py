"""
The :mod:`tensorly.decomposition` module includes utilities for performing
tensor decomposition such as CANDECOMP-PARAFAC and Tucker.                                                                                               
"""

from .candecomp_parafac import parafac, non_negative_parafac, randomised_parafac, sample_khatri_rao
from ._tucker import tucker, partial_tucker, non_negative_tucker
from .robust_decomposition import robust_pca
from .mps_decomposition import matrix_product_state
from .cmtf_als import align_tensors, factor_match_score_3d, coupled_matrix_tensor_3d_factorization
from .parafac2 import parafac2
from .symmetric_parafac import symmetric_parafac_power_iteration, symmetric_power_iteration
