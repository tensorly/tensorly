=============
API reference
=============

:mod:`tensorly`: Manipulating the backend with a unified interface
==================================================================

For each backend, tensorly provides the following uniform functions:

.. automodule:: tensorly
    :no-members:
    :no-inherited-members:

.. autosummary::
    :toctree: generated
    :template: function.rst

    set_backend
    get_backend
    context
    tensor
    is_tensor
    shape
    ndim
    to_numpy
    copy
    concatenate
    reshape
    transpose
    moveaxis
    arange
    ones
    zeros
    zeros_like
    eye
    where
    clip
    max
    min
    all
    mean
    sum
    prod
    sign
    abs
    sqrt
    norm
    dot
    kron
    solve
    qr
    kr
    partial_svd


:mod:`tensorly.base`: Core tensor functions
============================================

.. automodule:: tensorly.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.base

.. autosummary::
    :toctree: generated/
    :template: function.rst

    unfold
    fold
    tensor_to_vec
    vec_to_tensor
    partial_unfold
    partial_fold
    partial_tensor_to_vec
    partial_vec_to_tensor


:mod:`tensorly.kruskal_tensor`: Tensors in the Kruskal format
=============================================================

.. automodule:: tensorly.kruskal_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.kruskal_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    kruskal_to_tensor
    kruskal_to_unfolded
    kruskal_to_vec
    kruskal_mode_dot
    unfolding_dot_khatri_rao


:mod:`tensorly.tucker_tensor`: Tensors in Tucker format
=======================================================

.. automodule:: tensorly.tucker_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tucker_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tucker_to_tensor
    tucker_to_unfolded
    tucker_to_vec
    tucker_mode_dot


:mod:`tensorly.mps_tensor`: Tensors in Matrix-Product-State format
==================================================================

.. automodule:: tensorly.mps_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.mps_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mps_to_tensor
    mps_to_unfolded
    mps_to_vec


:mod:`tensorly.tenalg`: Tensor algebra
======================================

.. automodule:: tensorly.tenalg
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tenalg

.. autosummary::
    :toctree: generated/
    :template: function.rst

    khatri_rao
    kronecker
    mode_dot
    multi_mode_dot
    proximal.soft_thresholding
    proximal.svd_thresholding
    proximal.procrustes
    inner
    contract


:mod:`tensorly.decomposition`: Tensor Decomposition
====================================================

.. automodule:: tensorly.decomposition
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.decomposition

.. autosummary::
    :toctree: generated/
    :template: function.rst

    parafac
    non_negative_parafac
    sample_khatri_rao
    randomised_parafac
    tucker
    partial_tucker
    non_negative_tucker
    robust_pca
    matrix_product_state


:mod:`tensorly.regression`: Tensor Regression
==============================================

.. automodule:: tensorly.regression
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.regression

.. autosummary::
    :toctree: generated/
    :template: class.rst

    tucker_regression.TuckerRegressor
    kruskal_regression.KruskalRegressor


:mod:`tensorly.metrics`: Performance measures
==============================================

.. automodule:: tensorly.metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.metrics

.. autosummary::
    :toctree: generated/
    :template: function.rst

    regression.MSE
    regression.RMSE


:mod:`tensorly.random`: Sampling random tensors
===============================================

.. automodule:: tensorly.random
   :no-members:
   :no-inherited-members:

.. currentmodule:: tensorly.random

.. autosummary::
   :toctree: generated/
   :template: function.rst

   random_kruskal
   random_tucker
   random_mps 
   check_random_state



:mod:`tensorly.datasets`: Creating and loading data
====================================================

.. automodule:: tensorly.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.datasets

.. autosummary::
    :toctree: generated/
    :template: function.rst

    synthetic.gen_image


:mod:`tensorly.contrib`: Experimental features
==============================================

.. automodule:: tensorly.contrib
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.contrib

.. autosummary::
    :toctree: generated/
    :template: function.rst

    decomposition.matrix_product_state_cross

Sparse tensor operations
------------------------

Enables tensor operations on sparse tensors.
Currently, the following decomposition methods are supported (for the NumPy backend, using Sparse):

.. automodule:: tensorly.contrib.sparse

.. currentmodule:: tensorly.contrib

.. autosummary::
    :toctree: generated/

   sparse.decomposition.tucker
   sparse.decomposition.partial_tucker
   sparse.decomposition.non_negative_tucker
   sparse.decomposition.robust_pca
   sparse.decomposition.parafac
   sparse.decomposition.non_negative_parafac


