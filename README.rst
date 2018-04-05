.. image:: https://badge.fury.io/py/tensorly.svg
    :target: https://badge.fury.io/py/tensorly
    
.. image:: https://anaconda.org/tensorly/tensorly/badges/version.svg   
    :target: https://anaconda.org/tensorly/tensorly

.. image:: https://travis-ci.org/tensorly/tensorly.svg?branch=master
    :target: https://travis-ci.org/tensorly/tensorly

.. image:: https://coveralls.io/repos/github/tensorly/tensorly/badge.svg?branch=master
    :target: https://coveralls.io/github/tensorly/tensorly?branch=master
    
.. image:: https://badges.gitter.im/tensorly/tensorly.svg
    :target: https://gitter.im/tensorly/tensorly?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge

========
TensorLy
========

TensorLy is a fast and simple Python library for tensor learning. It builds on top of NumPy, SciPy and MXNet and allows for fast and straightforward tensor decomposition, tensor learning and tensor algebra.

- **Website:** http://tensorly.github.io
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


