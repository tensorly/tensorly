"""
Load example datasets.
"""

from os.path import dirname
import numpy as np

class Bunch(dict):
    """ A Bunch, exposing dict keys as a keys() meathod.
    Definition from scikit-learn. """
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def IL2data():
    """ Loads tensor of IL-2 mutein treatment responses. """
    path_here = dirname(__file__)
    tensor = np.load(path_here + "/data/IL2_Response_Tensor.npy")
    ligands = ['IL2 (Mono)', 'IL15 (Mono)', 'WT N-term (Biv)', 'H16N N-term (Biv)',
                  'R38Q N-term (Biv)', 'R38Q/H16N (Biv)', 'WT N-term (Mono)',
                  'H16N N-term (Mono)', 'WT C-term (Mono)', 'V91K C-term (Mono)',
                  'R38Q N-term (Mono)', 'F42Q N-Term (Mono)', 'N88D C-term (Mono)']
    cells = ['Treg IL2Ra_hi', 'Treg', 'Treg IL2Ra_lo', 'Thelper IL2Ra_hi', 'Thelper', 'Thelper IL2Ra_lo', 'CD8+', 'NK']
    times = [4, 2, 1, 0.5]
    doses = np.geomspace(84.0, 0.000474, 12)
    dims = ["ligands", "times", "doses", "cells"]
    reference = ["Orcutt-Jahns, B., Emmel, P. C., Snyder, E. M., Posner, C., \
                Carlson, S. M., & Meyer, A. S. (2021). Multivalency enhances the \
                specificity of Fc-cytokine fusions. BioRxiv."]
    task = ["Non-Negative CP"]

    return Bunch(tensor=tensor, ligands=ligands, cells=cells, times=times, doses=doses, dims=dims, reference=reference, task=task)
