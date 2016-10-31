=============
API reference
=============

:mod:`tensorly.base`: Core tensor functions
=============================================

.. automodule:: tensorly.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.base

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tensor_from_frontal_slices
    unfold
    fold
    tensor_to_vec
    vec_to_tensor
    partial_unfold
    partial_fold
    partial_tensor_to_vec
    partial_vec_to_tensor


:mod:`tensorly.kruskal`: Kruskal tensors
==========================================

.. automodule:: tensorly.kruskal
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.kruskal

.. autosummary::
    :toctree: generated/
    :template: function.rst

    kruskal_to_tensor
    kruskal_to_unfolded
    kruskal_to_vec

:mod:`tensorly.tucker`: Tucker tensors
========================================

.. automodule:: tensorly.tucker
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tucker

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
    norm
    higher_order_moment
    multi_mode_dot
    proximal.soft_thresholding
    proximal.inplace_soft_thresholding
    proximal.svd_thresholding
    proximal.procrustes
    partial_svd


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
    non_negative_tucker


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


:mod:`tensorly.utils`: Utility functions
=========================================

.. automodule:: tensorly.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.utils

.. autosummary::
    :toctree: generated/

    check_random_state
