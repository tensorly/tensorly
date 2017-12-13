.. image:: https://badge.fury.io/py/tensorly.svg
    :target: https://badge.fury.io/py/tensorly

.. image:: https://travis-ci.org/tensorly/tensorly.svg?branch=master
    :target: https://travis-ci.org/tensorly/tensorly

.. image:: https://coveralls.io/repos/github/tensorly/tensorly/badge.svg?branch=master
    :target: https://coveralls.io/github/tensorly/tensorly?branch=master
    
.. image:: https://badges.gitter.im/tensorly/tensorly.svg
    :target: https://gitter.im/tensorly/tensorly?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge

TensorLy
========

TensorLy is a fast and simple Python library for tensor learning. It builds on top of NumPy, SciPy and MXNet and allows for fast and straightforward tensor decomposition, tensor learning and tensor algebra.

- **Website:** http://tensorly.github.io
- **Source:**  https://github.com/tensorly/tensorly
- **Jupyter Notebooks:** https://github.com/JeanKossaifi/tensorly-notebooks


How to install
--------------
 
Easy option: install with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply run::

   pip install -U tensorly

That's it!

Alternatively, you can pip install from the git repository::

   pip install git+https://github.com/tensorly/tensorly

Development: install from git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library is still very new and under heavy developement. To install the last version:

Clone the repository and cd there::

   git clone https://github.com/tensorly/tensorly
   cd tensorly

Then install the package (here in editable mode with `-e` or equivalently `--editable`)::

   pip install -e .

Running the tests
~~~~~~~~~~~~~~~~~

Testing and documentation are an essential part of this package and all functions come with uni-tests and documentation.

You can run all the tests using the `nose` package::

   nosetests -v tensorly

