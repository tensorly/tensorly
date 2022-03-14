"""
Non-negative PARAFAC Decomposition of IL-2 Response Data
=========================================================

Here we will provide an example of how to use non-negative PARAFAC tensor 
decomposition (:func:`tensorly.decomposition.parafac`) to first reduce the dimensionality 
of a tensor of experimental data, and then make insights about the underlying structure 
of that data.

To do this, we will work with a tensor of experimentally measured cell signaling data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorly.datasets import IL2data
from tensorly.decomposition import non_negative_parafac
from tensorly.cp_tensor import cp_normalize

#%%
# Here we will load a tensor of experimentally measured cellular responses to 
# IL-2 stimulation. IL-2 is a naturally occurring immune signaling molecule 
# which has been engineered by pharmaceutical companies and drug designers 
# in attempts to act as an effective immunotherapy. In order to make effective IL-2
# therapies, pharmaceutical engineer have altered IL-2's signaling activity in order to
# increase or decrease its interactions with particular cell types. 
# 
# IL-2 signals through the Jak/STAT pathway and transmits a signal into immune cells by 
# phosphorylating STAT5 (pSTAT5). When phosphorylated, STAT5 will cause various immune  
# cell types to proliferate, and depending on whether regulatory (regulatory T cells, or Tregs) 
# or effector cells (helper T cells, natural killer cells, and cytotoxic T cells,
# or Thelpers, NKs, and CD8+ cells) respond, IL-2 signaling can result in 
# immunosuppression or immunostimulation respectively. Thus, when designing a drug
# meant to repress the immune system, potentially for the treatment of autoimmune
# diseases, IL-2 which primarily enacts a response in Tregs is desirable. Conversely,
# when designing a drug that is meant to stimulate the immune system, potentially for
# the treatment of cancer, IL-2 which primarily enacts a response in effector cells
# is desirable. In order to achieve either signaling bias, IL-2 variants with altered
# affinity for it's various receptors (IL2Rα or IL2Rβ) have been designed. Furthermore
# IL-2 variants with multiple binding domains have been designed as multivalent 
# IL-2 may act as a more effective therapeutic. In order to understand how these mutations
# and alterations affect which cells respond to an IL-2 mutant, we will perform 
# non-negative PARAFAC tensor decomposition on our cell response data tensor.
# 
# Here, our data contains the responses of 8 different cell types to 13 different 
# IL-2 mutants, at 4 different timepoints, at 12 standardized IL-2 concentrations.
# Therefore, our tensor will have shape (13 x 4 x 12 x 8), with dimensions
# representing IL-2 mutant, stimulation time, dose, and cell type respectively. Each
# measured quantity represents the amount of phosphorlyated STAT5 (pSTAT5) in a 
# given cell population following stimulation with the specified IL-2 mutant.

response_data = IL2data()
IL2mutants, cells = response_data.ticks[0], response_data.ticks[3]
print(response_data.tensor.shape, response_data.dims)

#%%
# Now we will run non-negative PARAFAC tensor decomposition to reduce the dimensionality 
# of our tensor. We will use 3 components, and normalize our resulting tensor to aid in 
# future comparisons of correlations across components.
#
# First we must preprocess our tensor to ready it for factorization. Our data has a 
# few missing values, and so we must first generate a mask to mark where those values
# occur.

tensor_mask = np.isfinite(response_data.tensor)

#%%
# Now that we've marked where those non-finite values occur, we can regenerate our 
# tensor without including non-finite values, allowing it to be factorized.

response_data_fin = np.nan_to_num(response_data.tensor)

#%%
# Using this mask, and finite-value only tensor, we can decompose our signaling data into
# three components. We will also normalize this tensor, which will allow for easier
# comparisons to be made between the meanings, and magnitudes of our resulting components.

sig_tensor_fact = non_negative_parafac(response_data_fin, init='random', rank=3, mask=tensor_mask, n_iter_max=5000, tol=1e-9, random_state=1)
sig_tensor_fact = cp_normalize(sig_tensor_fact)

#%%
# Now we will load the names of our cell types and IL-2 mutants, in the order in which 
# they are present in our original tensor. IL-2 mutant names refer to the specific 
# mutations made to their amino acid sequence, as well as their valency 
# format (monovalent or bivalent).
# 
# Finally, we label, plot, and analyze our factored tensor of data.

f, ax = plt.subplots(1, 2, figsize=(12, 6))

lig_facs = sig_tensor_fact[1][0]
lig_fact_df = pd.DataFrame()
for i in range(0, 3):
    lig_fact_df = lig_fact_df.append(pd.DataFrame({"Component_Val": lig_facs[:, i], "Component": (i + 1), "Ligand": IL2mutants}))

sns.barplot(data=lig_fact_df, ax=ax[0], x="Component_Val", y="Ligand", hue="Component")
ax[0].set(xlabel="Component Weight", xlim=(0, 1))

cell_facs = sig_tensor_fact[1][3]
cell_fact_df = pd.DataFrame()
for i in range(0, 3):
    cell_fact_df = cell_fact_df.append(pd.DataFrame({"Component_Val": cell_facs[:, i], "Component": (i + 1), "Cell": cells}))

sns.barplot(data=cell_fact_df, ax=ax[1], x="Cell", y="Component_Val", hue="Component")
ax[1].set(ylabel="Component Weight", ylim=(0, 1))
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right")

plt.show()

#%%
# Here we observe the correlations which both ligands and cell types have with each of 
# our three components - we can interepret our tensor factorization for looking for 
# patterns among these correlations. 
# 
# For example, we can see that bivalent mutants generally have higher correlations with
# component two, as do regulatory T cells. Thus we can infer that bivalent ligands 
# activate regulatory T cells more than monovalent ligands. We also see that this 
# relationship is strengthened by the availability of IL2Rα, one subunit of the IL-2 receptor.
#
# This is just one example of an insight we can make using tensor factorization. 
# By plotting the correlations which time and dose have with each component, we 
# could additionally make inferences as to the dynamics and dose dependence of how mutations 
# affect IL-2 signaling in immune cells.
