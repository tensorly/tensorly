"""
Load example datasets.
"""

from os.path import dirname
import numpy as np


class Bunch(dict):
    """ A Bunch, exposing dict keys as a keys() method.
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
    """
    Loads tensor of IL-2 mutein treatment responses.
    Tensor contains the signaling responses of eight different cell types to 13 IL-2 mutants.
    Dose-responses to 12 concentrations of muteins were collected at four time points for each mutant and cell type.
    Ligands include IL-2 mutants with modified receptor affinities, and in both mono- and bivalent formats.
    Useful for demonstrating execution and interpretation of non-negative CP decomposition.
    """
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
    DESC = "This dataset includes a tensor of eight cell types' responses to 13 IL-2 muteins, at 4 different time points, and at 12 different IL-2 concentrations."
    LICENSE = """Copyright 2021 The Regents of the University of California

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE."""

    return Bunch(
        tensor=tensor,
        ligands=ligands,
        cells=cells,
        times=times,
        doses=doses,
        dims=dims,
        reference=reference,
        task=task,
        DESC=DESC,
        LICENSE=LICENSE)


def COVID19_data():
    """
    Load an example dataset of COVID-19 systems serology.
    Formatted in a three-mode tensor of samples, antigens, and receptors
    """

    path_here = dirname(__file__)
    tensor = np.load(path_here + "/data/COVID19_data.npy")

    sampleLabels = np.array(['Negative', 'Negative', 'Negative', 'Negative', 'Negative',
                             'Negative', 'Negative', 'Negative', 'Negative', 'Negative',
                             'Negative', 'Negative', 'Negative', 'Negative', 'Negative',
                             'Negative', 'Negative', 'Negative', 'Negative', 'Negative',
                             'Negative', 'Negative', 'Negative', 'Negative', 'Negative',
                             'Negative', 'Negative', 'Negative', 'Negative', 'Negative',
                             'Negative', 'Negative', 'Negative', 'Negative', 'Negative',
                             'Negative', 'Negative', 'Negative', 'Negative', 'Mild', 'Mild',
                             'Mild', 'Mild', 'Mild', 'Mild', 'Mild', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Severe', 'Severe',
                             'Severe', 'Severe', 'Severe', 'Severe', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased', 'Deceased', 'Deceased', 'Deceased',
                             'Deceased', 'Deceased'], dtype=object)
    antigenLabels = ["S", "RBD", "N", "S1", "S2", "S1 Trimer"]
    receptorLabels = ["IgG1", "IgG2", "IgG3", "IgA1", "IgA2", "IgM", "FcRalpha", "FcR2A", "FcR2B", "FcR3A", "FcR3B"]

    dims = ["samples", "antigens", "receptors"]
    reference = ["Tan, Z. C., Murphy, M. C., Alpay, H. S., Taylor, S. D., & Meyer, A. S. (2021). Tensor‐structured decomposition improves systems serology analysis.  Molecular systems biology, 17(9), e10243.",
                 "Zohar, T., Loos, C., Fischinger, S., Atyeo, C., Wang, C., Slein, M. D., ... & Alter, G. (2020). Compromised humoral functional evolution tracks with SARS-CoV-2 mortality. Cell, 183(6), 1508-1519."]
    task = ["CP decomposition demonstration"]
    DESC = "An example dataset of COVID-19 systems serology."
    LICENSE = """Copyright 2022 The Regents of the University of California

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE."""

    return Bunch(
        tensor=tensor,
        ticks=[sampleLabels, antigenLabels, receptorLabels],
        dims=dims,
        reference=reference,
        task=task,
        DESC=DESC,
        LICENSE=LICENSE)