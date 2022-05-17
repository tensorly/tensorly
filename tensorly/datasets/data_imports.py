"""
Load example datasets.
"""

from os.path import dirname
import numpy as np
import requests
import scipy.io
from zipfile import ZipFile
from io import BytesIO


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
        ticks=[ligands, times, doses, cells],
        dims=dims,
        reference=reference,
        task=task,
        DESC=DESC,
        LICENSE=LICENSE)


def load_covid19_serology():
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
    DESC = """ A demonstration on applying CP decomposition to a COVID-19 systems serology dataset.
            Here, serum antibodies of 438 samples were systematically profiled by their binding behavior
            to SARS-CoV-2 (the virus that causes COVID-19) antigens and Fc receptors (a type of receptors 
            that expressed by immune cells) activities. Samples are labeled by the status of the patients.
            """
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
        DESC=DESC,
        LICENSE=LICENSE)


def load_indian_pines():
    """
    Loads indian pines hyperspectral data from th website and returns it as a tensorly tensor without storing the data
    in the hard drive. This dataset could be useful for non-negative constrained decomposition methods and
    classification/segmentation applications with tha available ground truth in
    http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat.
    """

    url = 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
    r = requests.get(url, allow_redirects=True)
    image = scipy.io.loadmat(BytesIO(r.content))['indian_pines_corrected']
    reference = "Baumgardner, M. F., Biehl, L. L., Landgrebe, D. A. (2015). 220 Band AVIRIS Hyperspectral " \
                "Image Data Set: June 12, 1992 Indian Pine Test Site 3. Purdue University Research Repository. " \
                "doi:10.4231/R7RX991C"
    licence = "Licensed under Attribution 3.0 Unported (https://creativecommons.org/licenses/by/3.0/legalcode)"
    desc =  "Airborne Visible / Infrared Imaging Spectrometer (AVIRIS)  hyperspectral sensor data (aviris.jpl.nasa.gov/) " \
            "were acquired on June 12, 1992 over the Purdue University Agronomy farm northwest " \
            "of West Lafayette and the surrounding area. This scene consists of 145 times 145 pixels and 220 spectral " \
            "reflectance bands in the wavelength range 0.4–2.5 10^(-6) meters."
    url_gt = 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
    r = requests.get(url_gt, allow_redirects=True)
    labels = scipy.io.loadmat(BytesIO(r.content))['indian_pines_gt']
    wavelenghts = []
    return Bunch(
        tensor=np.array(image, "float"),
        ticks=[labels, wavelenghts],
        dims=["Spatial dimension", "Spatial dimension", "Hyperspectral bands"],
        reference=reference,
        DESC=desc,
        LICENCE=licence)


def load_kinetic():
    """
    Loads kinetic fluorescence dataset from website and returns it as tensorly tensor without storing the data
    in the hard drive.The data is well suited for Parafac and multi-way partial least squares regression (N-PLS).
    """
    url = 'http://models.life.ku.dk/sites/default/files/Kinetic_Fluor.zip'
    r = requests.get(url, allow_redirects=True)
    zip = ZipFile(BytesIO(r.content))
    tensor = scipy.io.loadmat(zip.open('Xlarge.mat'))['Xlarge']
    tensor[np.isnan(tensor)] = 0
    reference = "Nikolajsen, R. P., Booksh, K. S., Hansen, Å. M., & Bro, R. (2003). \
                Quantifying catecholamines using multi-way kinetic modelling. \
                Analytica Chimica Acta, 475(1-2), 137-150."
    licence = "http://www.models.life.ku.dk/datasets. All downloadable material listed on these pages - " \
              "appended by specifics mentioned under " \
              "the individual headers/chapters - is available for public use. " \
              "Please note that while great care has been taken, the software, code and data are provided" \
              "as is and that Q&T, LIFE, KU does not accept any responsibility or liability."
    desc = "A four-way data set with the modes: Concentration, excitation wavelength, emission wavelength and time"

    return Bunch(
        tensor=tensor,
        dims=["Measurements", "Emissions", "Excitations", "Time points"],
        reference=reference,
        DESC = desc,
        LICENCE = licence)
