=============
API reference
=============

:mod:`tensorly`: Setting the backend
====================================

.. automodule:: tensorly
    :no-members:
    :no-inherited-members:

.. autosummary:: 
    :toctree: generated
    :template: function.rst

    set_backend
    get_backend
    context
    to_numpy

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


:mod:`tensorly.kruskal`: Tensors in the Kruskal format
======================================================

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

:mod:`tensorly.tucker`: Tensors in Tucker format
================================================

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


:mod:`tensorly.tenalg`: Tensor algebra
=============================================

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
    tucker
    partial_tucker
    non_negative_tucker
    robust_pca


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

   cp_tensor
   tucker_tensor
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

