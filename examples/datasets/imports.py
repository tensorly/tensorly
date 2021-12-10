"""
Load example datasets.
"""

import numpy as np


def IL2data():
    return np.load("datasets/IL2_Response_Tensor.npy", allow_pickle=True)
