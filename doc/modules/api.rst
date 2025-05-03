=============
API reference
=============

Unified backend interface (:mod:`tensorly`)
===========================================

There are several libraries for multi-dimensional array computation, including NumPy, PyTorch, TensorFlow, JAX, CuPy and Paddle.
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

You can simply call ``set_backend('pytorch')`` to set the backend to `PyTorch`, and similarly for NumPy, JAX, etc. 
You can also use the context manager ``backend_context`` if you want to execute a block of code with a different backend.

.. autosummary::
    :toctree: generated
    :template: function.rst

    set_backend
    get_backend
    backend_context
    use_dynamic_dispatch
    use_static_dispatch

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


Core functions (:mod:`tensorly.base`)
=====================================

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


Tensors in CP form (:mod:`tensorly.cp_tensor`)
==============================================

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
    cp_permute_factors


Tensors in Tucker form (:mod:`tensorly.tucker_tensor`)
======================================================

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


Tensors in TT (MPS) form (:mod:`tensorly.tt_tensor`)
====================================================

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


Matrices in TT form (:mod:`tensorly.tt_matrix`)
===============================================

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


Tensors in PARAFAC2 form (:mod:`tensorly.parafac2_tensor`)
==========================================================

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


Tensor Algebra (:mod:`tensorly.tenalg`)
=======================================

Available functions
-------------------

TensorLy provides you with all the tensor algebra functions you need:

.. automodule:: tensorly.tenalg
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.tenalg

A unified SVD interface: 

.. autosummary::
    :toctree: generated/
    :template: function.rst

    svd_interface

Other tensor algebraic functionalities:

.. autosummary::
    :toctree: generated/
    :template: function.rst

    khatri_rao
    unfolding_dot_khatri_rao
    kronecker
    mode_dot
    multi_mode_dot
    inner
    outer
    batched_outer
    tensordot
    higher_order_moment
    proximal.soft_thresholding
    proximal.svd_thresholding
    proximal.procrustes

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


Tensor Decomposition (:mod:`tensorly.decomposition`)
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
    TensorTrain
    TensorRing
    TensorTrainMatrix


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
    tensor_ring
    parafac2
    constrained_parafac


Preprocessing (:mod:`tensorly.preprocessing`)
=============================================

.. automodule:: tensorly.preprocessing
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.preprocessing

.. autosummary::
    :toctree: generated/
    :template: function.rst

    svd_compress_tensor_slices
    svd_decompress_parafac2_tensor


Tensor Regression (:mod:`tensorly.regression`)
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
    CP_PLSR


Solvers (:mod:`tensorly.solvers`)
=================================

Tensorly provides with efficient solvers for nonnegative least squares problems which are crucial to nonnegative tensor decomposition, as well as a generic admm solver useful for constrained decompositions. Several proximal (projection) operators are located in tenalg.

.. automodule:: tensorly.solvers
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.solvers

.. autosummary::
    :toctree: generated/
    :template: function.rst

    nnls.hals_nnls
    nnls.fista
    nnls.active_set_nnls
    admm.admm

Performance measures (:mod:`tensorly.metrics`)
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
    correlation_index


Sampling tensors (:mod:`tensorly.random`)
=========================================

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



Datasets (:mod:`tensorly.datasets`)
===================================

.. automodule:: tensorly.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.datasets

.. autosummary::
    :toctree: generated/
    :template: function.rst

    synthetic.gen_image
    load_IL2data
    load_covid19_serology
    load_indian_pines
    load_kinetic

Plugin functionalities (:mod:`tensorly.plugins`)
================================================
Automatically cache the optimal contraction path when using the `einsum` tensor algebra backend

.. automodule:: tensorly.plugins
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.plugins

.. autosummary::
    :toctree: generated/
    :template: function.rst

    use_opt_einsum
    use_default_einsum
    use_cuquantum


Experimental features (:mod:`tensorly.contrib`)
===============================================

.. automodule:: tensorly.contrib
    :no-members:
    :no-inherited-members:

.. currentmodule:: tensorly.contrib

.. autosummary::
    :toctree: generated/
    :template: function.rst

    decomposition.tensor_train_cross
    decomposition.tensor_train_OI

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
