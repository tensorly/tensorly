.. raw:: html

    <p align="center"><img width="35%" src="doc/_static/TensorLy_logo.png" /></p>
    
    <p align="center">
        <a href="https://badge.fury.io/py/tensorly" target=blank>
            <img src="https://badge.fury.io/py/tensorly.svg"
        </a>
        <a href="https://anaconda.org/tensorly/tensorly" target=blank>
            <img src="https://anaconda.org/tensorly/tensorly/badges/version.svg" 
        </a>
        <a href="https://travis-ci.org/tensorly/tensorly" target=blank>
            <img src="https://travis-ci.org/tensorly/tensorly.svg?branch=master" 
        </a>
        <a href="https://coveralls.io/github/tensorly/tensorly?branch=master" target=blank>
            <img src="https://coveralls.io/repos/github/tensorly/tensorly/badge.svg?branch=master" 
        </a>
        <a href="https://gitter.im/tensorly/tensorly?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge" target=blank>
            <img src="https://badges.gitter.im/tensorly/tensorly.svg"
        </a>
    </p>
    
\ 

========
TensorLy
========


TensorLy is a Python library that aims at making tensor learning simple and accessible. It allows to easily perform tensor decomposition, tensor learning and tensor algebra. Its backend system allows to seamlessly perform computation with NumPy, MXNet or PyTorch and run methods at scale on CPU or GPU.

- **Website:** http://tensorly.org
- **Source-code:**  https://github.com/tensorly/tensorly
- **Jupyter Notebooks:** https://github.com/JeanKossaifi/tensorly-notebooks

----------------------------

Installing TensorLy
===================



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
|             git clone https://github.com/tensorly/tensorly                                    |
|             cd tensorly                                                                       |
|             # Install in editable mode with `-e` or, equivalently, `--editable`               |
|             pip install -e .                                                                  |
|                                                                                               |
+-----------------------------------------------------------------------------------------------+  
 
**Note:** TensorLy depends on NumPy by default. If you want to use the MXNet or PyTorch backends, you will need to install these packages separately.

For detailed instruction, checkout the `documentation <https://tensorly.github.io/stable/installation.html>`_.

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

Quickstart
==========

.. code:: python

   import tensorly as tl
   import numpy as np

Create a small third order tensor of size 3 x 4 x 2 and perform simple operations on it:

.. code:: python

   tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)))
   tl.unfolded = unfold(tensor, mode=0)
   tl.fold(unfolded, mode=0, shape=tensor.shape)


Applying tensor decomposition is easy:

.. code:: python

   from tensorly.decomposition import tucker
   # Apply Tucker decomposition 
   core, factors = tucker(tensor, rank=[2, 2, 2])
   # Reconstruct the full tensor from the decomposed form
   tl.tucker_to_tensor(core, factors) 

Changing the backend to perform computation on GPU for instance. Note that using MXNet or PyTorch requires to have installed them first:

.. code:: python

   tl.set_backend('pytorch') # Or 'mxnet' or 'numpy'

Now all the computation is done by PyTorch, and tensors can be created on GPU:

.. code:: python

   import torch
   tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=torch.cuda.FloatTensor)
   type(tensor) # torch.cuda.FloatTensor

For more information on getting started, checkout the `user-guide <https://tensorly.github.io/dev/user_guide/index.html>`_  and for a detailed reference of the functions and their documentation, refer to
the `API <https://tensorly.github.io/dev/modules/api.html>`_   

If you see a bug, open an `issue <https://github.com/tensorly/tensorly/issues>`_, or better yet, a `pull-request <https://github.com/tensorly/tensorly/pulls>`_!

-------------
