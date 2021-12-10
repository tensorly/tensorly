"""
Load example datasets.
"""

import numpy as np


def IL2data():
    IL2mutants = ['IL2 (Mono)', 'IL15 (Mono)', 'WT N-term (Biv)', 'H16N N-term (Biv)',
                  'R38Q N-term (Biv)', 'R38Q/H16N (Biv)', 'WT N-term (Mono)',
                  'H16N N-term (Mono)', 'WT C-term (Mono)', 'V91K C-term (Mono)',
                  'R38Q N-term (Mono)', 'F42Q N-Term (Mono)', 'N88D C-term (Mono)']

    cells = ['Treg IL2Ra_hi', 'Treg', 'Treg IL2Ra_lo', 'Thelper IL2Ra_hi', 'Thelper', 'Thelper IL2Ra_lo', 'CD8+', 'NK']

    return np.load("datasets/IL2_Response_Tensor.npy"), (IL2mutants, cells)
