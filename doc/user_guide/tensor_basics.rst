Tensor basics
=============

Creating a tensor
-----------------
A tensor is nothing more than a multi-dimensional array.

Let's take for this example the tensor :math:`\tilde X` defined by its frontal slices:

.. math::

   X_1 =
   \left[
   \begin{matrix}
   0  & 2  & 4  & 6\\
   8  & 10 & 12 & 14\\
   16 & 18 & 20 & 22\\
   \end{matrix}
   \right]

   \text{and}

   X_2 =
   \left[
   \begin{matrix}
   1  & 3  & 5  & 7\\
   9  & 11 & 13 & 15\\
   17 & 19 & 21 & 23\\
   \end{matrix}
   \right]


In Python, this array can be expressed as a numpy array::

   >>> import numpy as np
   >>> import tensorly as tl
   >>> X = tl.tensor(np.arange(24).reshape((3, 4, 2)))


You can view the frontal slices by fixing the last axis::

   >>> X[..., 0]
   array([[ 0,  2,  4,  6],
          [ 8, 10, 12, 14],
          [16, 18, 20, 22]])

   >>> X[..., 1]
   array([[ 1,  3,  5,  7],
          [ 9, 11, 13, 15],
          [17, 19, 21, 23]])


Unfolding
---------
Also called **matrization**, **unfolding** a tensor is done by reading the element in a given way as to obtain a matrix instead of a tensor.

For a tensor of size :math:`(I_0, I_1, \cdots, I_N)`, the n-mode unfolding of this tensor will be of size :math:`(I_n, I_0 \times I_1 \times \cdots \times I_{n-1} \times I_{n+1} \cdots \times I_N)`.

.. important::
   In TensorLy we use an unfolding different from the classical one as defined in [1]_ for better performance.

   Given a tensor :math:`\tilde X \in \mathbb{R}^{I_0, I_1 \times I_2 \times \cdots \times I_N}`, the
   mode-n unfolding of :math:`\tilde X` is a matrix :math:`\mathbf{X}_{[n]} \in \mathbb{R}^{I_n, I_M}`,
   with :math:`M = \prod_{\substack{k=0,\\k \neq n}}^N I_k` and is defined by
   the mapping from element :math:`(i_0, i_1, \cdots, i_N)` to :math:`(i_n, j)`, with

   .. math::
      j = \sum_{\substack{k=0,\\k \neq n}}^N i_k \times \prod_{\substack{m=k+1,\\ k \neq n}}^N I_m.


.. warning::

   Traditionally, mode-1 unfolding denotes the unfolding along the first dimension.
   However, to be consistent with the Python indexing that always starts at zero,
   in TensorLy, unfolding also starts at zero!

   Therefore ``unfold(tensor, 0)`` will unfold said tensor along its first dimension!


For instance, using the :math:`\tilde X` previously defined, the 0-mode unfolding of :math:`\tilde X`:

.. math::

   \tilde X_{[0]} =
   \left[ \begin{matrix}
      0  & 1  & 2  & 3  &  4 & 5  & 6  & 7\\
      8  & 9  & 10 & 11 & 12 & 13 & 14 & 15\\
      16 & 17 & 18 & 19 & 20 & 21 & 22 & 23\\
   \end{matrix} \right]


The 1-mode unfolding is given by:

.. math::

   \tilde X_{[1]} =
   \left[ \begin{matrix}
      0 & 1 & 8  & 9  & 16 & 17\\
      2 & 3 & 10 & 11 & 18 & 19\\
      4 & 5 & 12 & 13 & 20 & 21\\
      6 & 7 & 14 & 15 & 22 & 23\\
   \end{matrix} \right]

Finally, the 2-mode unfolding is the unfolding along the last axis:

.. math::

   \tilde X_{[2]} =
   \left[ \begin{matrix}
      0 & 2 & 4 & 6 & 8 & 10 & 12 & 14 & 16 & 18 & 20 & 22\\
      1 & 3 & 5 & 7 & 9 & 11 & 13 & 15 & 17 & 19 & 21 & 23\\
   \end{matrix} \right]


In TensorLy:

.. code-block:: python

   >>> from tensorly import unfold
   >>> unfold(X, 0) # mode-1 unfolding
   array([[ 0,  1,  2,  3,  4,  5,  6,  7],
          [ 8,  9, 10, 11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20, 21, 22, 23]])

   >>> unfold(X, 1) # mode-2 unfolding
   array([[ 0,  1,  8,  9, 16, 17],
          [ 2,  3, 10, 11, 18, 19],
          [ 4,  5, 12, 13, 20, 21],
          [ 6,  7, 14, 15, 22, 23]])

   >>> unfold(X, 2) # mode-3 unfolding
   array([[ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22],
          [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23]])

Folding
-------
You can **fold** an unfolded tensor back from matrix to full tensor using the :func:`tensorly.base.fold` function.

.. code-block:: python

   >>> from tensorly import fold
   >>> unfolding = unfold(X, 1)
   >>> original_shape = X.shape
   >>> fold(unfolding, 1, original_shape)
   array([[[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7]],

       [[ 8,  9],
        [10, 11],
        [12, 13],
        [14, 15]],

       [[16, 17],
        [18, 19],
        [20, 21],
        [22, 23]]])


References
----------
.. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
