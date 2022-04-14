=============
API reference
=============

:mod:`tensorly`: A unified backend interface
============================================

There are several libraries for multi-dimensional array computation, including NumPy, PyTorch, MXNet, TensorFlow, JAX and CuPy.
They all have strenghs and weaknesses, e.g. some are better on CPU, some better on GPU etc. 
Therefore, in TensorLy we enable you to use our algorithm (and any code you write using the library), with any of these libraries.

However, while they all loosely follow the API introduced and popularized by NumPy, there are differences. To make switching from one backend to another completely transparent, in TensorLy, we provide a thin wrapper to these libraries.
So instead of using PyTorch or NumPy functions (``pytorch.tensor`` or ``numpy.array`` for instance), 
you should only use functions through the backend (:func:`tensorly.tensor` in this case).

.. automodule:: tensorly
    :no-members:
    :no-inherited-members:

Setting the backend
-------------------

.. autosummary::
    :toctree: generated
    :template: function.rst

    set_backend
    get_backend
    backend_context

Context of a tensor
-------------------

In TensorLy, we provide some convenient functions to manipulate backend specific 
information on the tensors (the *context* of that tensor),
including dtype (e.g. `float32`, `float64`, etc), 
its *device* (e.g. CPU and GPU) where applicable, etc. 
We also provide functions to check if a tensor is on the current backend, convert to NumPy, etc.

.. autosummary::
   :toctree: generated
   :template: function.rst

   context
   is_tensor
   to_numpy
   eps
   finfo

Index assignement ("NumPy style")
---------------------------------

While in some backends (e.g. NumPy), you can directly combine indexing and assignement,
not all backends support this. Instead of 
``tensor[indices] = values``, you should use 
``tensor = tensorly.index_update(tensor, tensorly.index, values)``.

.. autosummary::
   :toctree: generated
   :template: function.rst

    index_update
    index

Available backend functions
---------------------------

For each backend, tensorly provides the following uniform functions:

Array creation
++++++++++++++

.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor
    ones
    zeros
    zeros_like
    eye
    diag
   check_random_state

Array manipulation
++++++++++++++++++

.. autosummary::
    :toctree: generated
    :template: function.rst

    shape
    ndim
    copy
    concatenate
    conj
    reshape
    transpose
    moveaxis
    arange
    where
    clip
    max
    min
    argmax
    argmin
    all
    mean
    sum
    prod
    sign
    abs
    sqrt
    norm
    stack
    sort

Algebraic operations
++++++++++++++++++++

.. autosummary::
    :toctree: generated
    :template: function.rst

    dot
    matmul
    tensordot
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


:mod:`tensorly.cp_tensor`: Tensors in CP form
=============================================

.. automodule:: tensorly.cp_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.cp_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    cp_to_tensor
    cp_to_unfolded
    cp_to_vec
    cp_normalize
    cp_norm
    cp_mode_dot
    unfolding_dot_khatri_rao


:mod:`tensorly.tucker_tensor`: Tensors in Tucker form
=====================================================

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


:mod:`tensorly.tt_tensor`: Tensors in Tensor-Train (MPS) form
=============================================================

.. automodule:: tensorly.tt_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tt_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tt_to_tensor
    tt_to_unfolded
    tt_to_vec
    pad_tt_rank


:mod:`tensorly.tt_matrix`: Matrices in TT form
==============================================

.. automodule:: tensorly.tt_matrix
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tt_matrix

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tt_matrix_to_tensor
    tt_matrix_to_unfolded
    tt_matrix_to_vec


:mod:`tensorly.parafac2_tensor`: Tensors in PARAFAC2 form
=========================================================

.. automodule:: tensorly.parafac2_tensor
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.parafac2_tensor

.. autosummary::
    :toctree: generated/
    :template: function.rst

    parafac2_to_tensor
    parafac2_to_slice
    parafac2_to_slices
    parafac2_to_unfolded
    parafac2_to_vec


:mod:`tensorly.tenalg`: Tensor Algebra
======================================

Available functions
-------------------

TensorLy provides you with all the tensor algebra functions you need:

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
    outer
    batched_outer
    tensordot
    higher_order_moment

Tensor Algebra Backend
----------------------

For advanced users, you may want to dispatch all the computation to `einsum` (if available)
instead of using our manually optimized functions. 
In TensorLy, we enable this very easily through our tensor algebra backend.
If you have your own library implementing tensor algebraic functions, you could even use it that way!

.. autosummary::
    :toctree: generated/
    :template: function.rst

    set_backend
    get_backend
    backend_context


:mod:`tensorly.decomposition`: Tensor Decomposition
====================================================

.. automodule:: tensorly.decomposition
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.decomposition

Classes
-------

Note that these are currently experimental and may change in the future.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CP
    RandomizedCP
    CPPower
    CP_NN_HALS
    Tucker
    TensorTrain
    Parafac2
    SymmetricCP
    ConstrainedCP

Functions
---------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    parafac
    power_iteration
    parafac_power_iteration
    symmetric_power_iteration
    symmetric_parafac_power_iteration
    non_negative_parafac
    non_negative_parafac_hals
    sample_khatri_rao
    randomised_parafac
    tucker
    partial_tucker
    non_negative_tucker
    non_negative_tucker_hals
    robust_pca
    tensor_train
    tensor_train_matrix
    parafac2
    constrained_parafac


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
    cp_regression.CPRegressor


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
    factors.congruence_coefficient


:mod:`tensorly.random`: Sampling tensors
========================================

.. automodule:: tensorly.random
   :no-members:
   :no-inherited-members:

.. currentmodule:: tensorly.random

.. autosummary::
   :toctree: generated/
   :template: function.rst

   random_cp
   random_tucker
   random_tt
   random_tt_matrix
   random_parafac2



:mod:`tensorly.datasets`: Datasets
==================================

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

    decomposition.tensor_train_cross

Sparse tensors
--------------

The :mod:`tensorly.contrib.sparse` module enables tensor operations on sparse tensors.
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
   sparse.decomposition.symmetric_parafac_power_iteration


