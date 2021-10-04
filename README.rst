.. image:: https://badge.fury.io/py/tensorly.svg
    :target: https://badge.fury.io/py/tensorly

.. image:: https://anaconda.org/tensorly/tensorly/badges/version.svg   
    :target: https://anaconda.org/tensorly/tensorly

.. image:: https://github.com/tensorly/tensorly/workflows/Test%20TensorLy/badge.svg
    :target: https://github.com/tensorly/tensorly/actions?query=workflow%3A%22Test+TensorLy%22

.. image:: https://codecov.io/gh/tensorly/tensorly/branch/master/graph/badge.svg?token=mnZ234sGSA
    :target: https://codecov.io/gh/tensorly/tensorly

.. image:: https://img.shields.io/badge/Slack-join-brightgreen
    :target: https://join.slack.com/t/tensorly/shared_invite/zt-wqnts2sk-wbiRX6ml~Xt6~GDYWRPFfg


========
TensorLy
========
   

TensorLy is a Python library that aims at making tensor learning simple and accessible. It allows to easily perform tensor decomposition, tensor learning and tensor algebra. Its backend system allows to seamlessly perform computation with NumPy, PyTorch, JAX, MXNet, TensorFlow or CuPy, and run methods at scale on CPU or GPU.

- **Website:** http://tensorly.org
- **Source-code:**  https://github.com/tensorly/tensorly
- **Jupyter Notebooks:** https://github.com/JeanKossaifi/tensorly-notebooks

----------------------------

Installing TensorLy
===================

The only pre-requisite is to have **Python 3** installed. The easiest way is via the `Anaconda distribution <https://www.anaconda.com/download/>`_.

+-------------------------------------------+---------------------------------------------------+
|      **With pip** (recommended)           |         **With conda**                            |
+-------------------------------------------+---------------------------------------------------+
|                                           |                                                   |
| .. code::                                 | .. code::                                         |
|                                           |                                                   |
|   pip install -U tensorly                 |   conda install -c tensorly tensorly              |
|                                           |                                                   |
|                                           |                                                   |
+-------------------------------------------+---------------------------------------------------+
|                               **Development (from git)**                                      |
+-------------------------------------------+---------------------------------------------------+
|                                                                                               |
|          .. code::                                                                            |
|                                                                                               |
|             # clone the repository                                                            |
|             git clone https://github.com/tensorly/tensorly                                    |
|             cd tensorly                                                                       |
|             # Install in editable mode with `-e` or, equivalently, `--editable`               |
|             pip install -e .                                                                  |
|                                                                                               |
+-----------------------------------------------------------------------------------------------+  
 
**Note:** TensorLy depends on NumPy by default. If you want to use the MXNet or PyTorch backends, you will need to install these packages separately.

For detailed instruction, please see the `documentation <http://tensorly.org/dev/installation.html>`_.

------------------

Quickstart
==========

Creating tensors
----------------

Create a small third order tensor of size 3 x 4 x 2, from a NumPy array and perform simple operations on it:

.. code:: python

   import tensorly as tl
   import numpy as np


   tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=tl.float64)
   unfolded = tl.unfold(tensor, mode=0)
   tl.fold(unfolded, mode=0, shape=tensor.shape)


You can also create random tensors:

.. code:: python

   from tensorly import random
   
   # A random tensor
   tensor = random.random_tensor((3, 4, 2))
   # A random CP tensor in factorized form
   cp_tensor = random.random_tensor(shape=(3, 4, 2), rank='same')

You can also create tensors in TT-format, Tucker, etc, see `random tensors <http://tensorly.org/stable/modules/api.html#module-tensorly.random>`_.

Setting the backend
-------------------

You can change the backend to perform computation with a different framework. By default, the backend is NumPy, but you can also perform the computation using  PyTorch, TensorFlow, MXNet, JAX or CuPy (requires to have installed them first). For instance, after setting the backend to PyTorch, all the computation is done by PyTorch, and tensors can be created on GPU:

.. code:: python

   tl.set_backend('pytorch') # Or 'mxnet', 'numpy', 'tensorflow', 'cupy' or 'jax'
   tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)), device='cuda:0')
   type(tensor) # torch.Tensor
   

Tensor decomposition
--------------------

Applying tensor decomposition is easy:

.. code:: python

   from tensorly.decomposition import tucker
   # Apply Tucker decomposition 
   tucker_tensor = tucker(tensor, rank=[2, 2, 2])
   # Reconstruct the full tensor from the decomposed form
   tl.tucker_to_tensor(tucker_tensor)
   
We have `many more decompositions <http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition>`_ available, be sure to check them out!

Next steps
----------
This is just a very quick introduction to some of the basic features of TensorLy. 
For more information on getting started, checkout the `user-guide <http://tensorly.org/dev/user_guide/index.html>`_  and for a detailed reference of the functions and their documentation, refer to
the `API <http://tensorly.org/dev/modules/api.html>`_   

If you see a bug, open an `issue <https://github.com/tensorly/tensorly/issues>`_, or better yet, a `pull-request <https://github.com/tensorly/tensorly/pulls>`_!
  
--------------------------

Running the tests
=================

Testing and documentation are an essential part of this package and all functions come with uni-tests and documentation.

The tests are ran using the `pytest` package (though you can also use `nose`). 
First install `pytest`::

    pip install pytest
    
Then to run the test, simply run, in the terminal:

.. code::

   pytest -v tensorly
   
Alternatively, you can specify for which backend you wish to run the tests:

.. code::
   
   TENSORLY_BACKEND='numpy' pytest -v tensorly
 

------------------

Citing
======

If you use TensorLy in an academic paper, please cite [1]_::

    @article{tensorly,
      author  = {Jean Kossaifi and Yannis Panagakis and Anima Anandkumar and Maja Pantic},
      title   = {TensorLy: Tensor Learning in Python},
      journal = {Journal of Machine Learning Research},
      year    = {2019},
      volume  = {20},
      number  = {26},
      pages   = {1-6},
      url     = {http://jmlr.org/papers/v20/18-277.html}
    }
    
    
.. [1] Jean Kossaifi, Yannis Panagakis, Anima Anandkumar and Maja Pantic, **TensorLy: Tensor Learning in Python**, *Journal of Machine Learning Research (JMLR)*, 2019, volume 20, number 26.
