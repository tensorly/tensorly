from ...decomposition import (tucker, partial_tucker, non_negative_tucker,
                              parafac, non_negative_parafac, robust_pca)
from .core import wrap

tucker = wrap(tucker)
partial_tucker = wrap(partial_tucker)
non_negative_tucker = wrap(non_negative_tucker)
robust_pca = wrap(robust_pca)
parafac = wrap(parafac)
non_negative_parafac = wrap(non_negative_parafac)