"""
Apply CP decomposition to COVID-19 Serology Dataset
"""

##############################################################################
# Introduction
# -----------------------
# PARAFAC (CP) decomposition is extremely useful in dimensionality reduction, allowing us
# to develop models that are both representative and compact while retaining crucial patterns
# between subjects. Here, we provide an example of how it can be applied to biomedical research.
#
# Systems serology is a new technology that examines the antibodies from a patient's serum, aiming
# to comprehensively profile the interactions between the antibodies and
# [Fc receptors](https://en.wikipedia.org/wiki/Fc_receptor) alongside other types of immunological
# and demographic data. Here, we will apply CP decomposition to a [COVID-19 system serology dataset]
# (https://www.sciencedirect.com/science/article/pii/S0092867420314598).
#
# Details of this analysis as well as more in-depth biological implication can be found in
# [this work](https://www.embopress.org/doi/full/10.15252/msb.202110243).
#
# We first import this dataset of a panel of COVID-19 patients:

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.datasets.imports import COVID19_data
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from itertools import groupby
import seaborn as sns
from scipy.stats import sem

data = COVID19_data()

##############################################################################
# Apply CP decomposition to this dataset with Tensorly
# -----------------------
# Now we apply CP decomposition to this dataset.

comps = np.arange(1, 7)
CMTFfacs = [parafac(data.tensor, cc, tol=1e-10, n_iter_max=1000,
                    linesearch=True, orthogonalise=2) for cc in comps]

##############################################################################
# To evaluate how well CP decomposition explains the variance in the dataset, we plot the percent
# variance reconstructed (R2X) for a range of ranks.

def reconstructed_variance(tFac, tIn=None):
    """ This function calculates the amount of variance captured (R2X) by the tensor method. """
    tMask = np.isfinite(tIn)
    vTop = np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
    vBottom = np.sum(np.square(np.nan_to_num(tIn)))
    return 1.0 - vTop / vBottom


fig1 = plt.figure()
ax1 = fig1.add_axes([0,0,1,1])
CMTFR2X = np.array([reconstructed_variance(f, data.tensor) for f in CMTFfacs])
print(CMTFR2X)
ax1.plot(comps, CMTFR2X, "bo")
ax1.set_ylabel("R2X")
ax1.set_xlabel("Number of Components")
ax1.set_xticks([x for x in comps])
ax1.set_xticklabels([x for x in comps])
ax1.set_ylim(0, 1)
ax1.set_xlim(0.0, np.amax(comps) + 0.5)

##############################################################################
# Build a Logistic Regression classifier to predict severe versus deceased COVID-19 patients
# -----------------------
# An important function of CP decomposition here is to reduce data while keeping significant
# information.

tfac = CMTFfacs[1]

# Ensure that factors are negative on at most one direction.
tfac.factors[1][:, 0] *= -1
tfac.factors[2][:, 0] *= -1

##############################################################################
# To evaluate the classifier, we plot a ROC-AUC curve

def calculateROCcurve(tfac):
    subjj = np.isin(data.ticks[0], ["Severe", "Deceased"])

    X = tfac.factors[0][subjj, :]
    y = pd.factorize(data.ticks[0][subjj])[0]
    aucs = []

    kf = KFold(n_splits=10, shuffle=True)
    outt = pd.DataFrame(columns=["fold", "FPR", "TPR"])
    for ii, (train, test) in enumerate(kf.split(X)):
        model = LogisticRegression().fit(X[train], y[train])
        y_score = model.predict_proba(X[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        aucs.append(roc_auc_score(y[test], y_score[:, 1]))
        outt = pd.concat([outt, pd.DataFrame(data={"fold": [ii+1] * len(fpr), "FPR": fpr, "TPR": tpr})])

    xs = pd.unique(outt["FPR"])
    ipl = pd.DataFrame(columns=["fold", "FPR", "TPR"])
    for ii in range(kf.n_splits):
        ys = np.interp(xs, outt.loc[outt["fold"]==(ii+1), "FPR"], outt.loc[outt["fold"]==(ii+1), "TPR"])
        ys[0] = 0
        ipl = pd.concat([ipl, pd.DataFrame(data={"fold": [(ii+1)] * len(xs), "FPR": xs, "TPR": ys})])

    return ipl, aucs

roc_df, auc = calculateROCcurve(tfac)
roc_sum = roc_df.groupby(['FPR'], as_index=False).agg({'TPR': ['mean', 'sem']})


fig2 = plt.figure()
ax2 = sns.lineplot(x=roc_sum["FPR"], y=roc_sum["TPR"]["mean"], color='b')
sns.lineplot(x=[0, 1], y=[0, 1], color="black", ax=ax2)
tprs_upper = np.minimum(roc_sum["TPR"]["mean"] + roc_sum["TPR"]["sem"], 1)
tprs_lower = np.maximum(roc_sum["TPR"]["mean"] - roc_sum["TPR"]["sem"], 0)
ax2.fill_between(roc_sum["FPR"], tprs_lower,
                   tprs_upper, color='grey', alpha=.2)
ax2.set_title("Severe vs. Deceased ROC (AUC={}±{})".format(np.around(np.mean(auc), decimals=2),
                                                             np.around(sem(auc), decimals=2)))


##############################################################################
# Inspect the biological insights from CP components
# -----------------------
# Eventually, we wish CP decomposition can bring insights to this data set. For example, in this
# case, revealing the underlying trend of COVID-19 serum-level immunity. To do this, we can inspect
# how each component looks like on weights


def component_plot(factors, xlabel, ylabel, plotLabel, ax, d=False):
    """ Creates heatmap plots for each input dimension by component. """
    scales = np.linalg.norm(factors, ord=np.inf, axis=0)
    factors /= scales
    if d:
        b = [list(g) for _, g in groupby(ylabel)]
        newLabels = []
        for i, c in enumerate(b):
            newLabels.append([x + "  " if i == len(c)//2 else "–" if i ==
                              0 or i == len(c) - 1 else "·" for (i, x) in enumerate(c)])

        newLabels = [item for sublist in newLabels for item in sublist]

        sns.heatmap(factors, cmap="PiYG", center=0,
                    xticklabels=xlabel, yticklabels=newLabels, ax=ax)
    else:
        sns.heatmap(factors, cmap="PiYG", center=0,
                    xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_title(plotLabel)

components = [str(ii + 1) for ii in range(tfac.rank)]

fig3, axes = plt.subplots(1, 3, figsize=(16,6))
component_plot(tfac.factors[0], components, list(data.ticks[0]), "Samples", axes[0], True)
component_plot(tfac.factors[1], components, data.ticks[1], "Antigens", axes[1])
component_plot(tfac.factors[2], components, data.ticks[2], "Receptors", axes[2])

##############################################################################
# From the results, we can see that serum COVID-19 immunity separates into two distinct signals,
# represented by two CP components: a clear acute response with IgG3, IgM, and IgA, and a long-term,
# IgG1-specific response. Samples from patients with different symptoms can be distinguish from
# these two components. This indicates that CP decomposition is a great tool to find these biologically
# significant signals.

##############################################################################
# References
# ----------
# [1] Tan, Z. C., Murphy, M. C., Alpay, H. S., Taylor, S. D., & Meyer, A. S. (2021). Tensor‐structured
# decomposition improves systems serology analysis. Molecular systems biology, 17(9), e10243.
# <https://www.embopress.org/doi/full/10.15252/msb.202110243>
#
# [2] Zohar, T., Loos, C., Fischinger, S., Atyeo, C., Wang, C., Slein, M. D., ... & Alter, G. (2020).
# Compromised humoral functional evolution tracks with SARS-CoV-2 mortality. Cell, 183(6), 1508-1519.
# <https://www.sciencedirect.com/science/article/pii/S0092867420314598>
