Tensor regression
=================

TensorLy also allows you to perform Tensor Regression.

Setting
-------

Tensor regression is available in the module :mod:`tensorly.regression`.

Given a series of :math:`N` tensor samples/observations, :math:`\tilde X_i, i={1, \cdots, N}`, and corresponding labels :math:`y_i, i={1, \cdots, N}`, we want to find the weight tensor :math:`\tilde W` such that, for each :math:`i={1, \cdots, N}`: 

.. math::

   y_i = \langle \tilde X_i, \tilde W \rangle

We additionally impose that :math:`\tilde W` be a rank-r CP decomposition (CP regression) or a rank :math:`(r_1, \cdots, r_N)`-Tucker decomposition (Tucker regression).
For a detailed explanation on tensor regression, please refer to [1]_.

TensorLy implements both types of tensor regression as scikit-learn-like estimators.

For instance, Krusal regression is available through the :class:`tensorly.regression.CPRegression` object. This implements a fit method that takes as parameters `X`, the data tensor whose first dimension is the number of samples, and `y`, the corresponding vector of labels.

Given a set of testing samples, you can use the predict method to obtain the corresponding predictions from the model.

References
----------
.. [1] W. Guo, I. Kotsia, and I. Patras. “Tensor Learning for Regression”,
       IEEE Transactions on Image Processing 21.2 (2012), pp. 816–827
