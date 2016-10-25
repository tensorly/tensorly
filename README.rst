.. -*- mode: rst -*-

.. image:: https://badge.fury.io/py/tensorly.svg
   :target: https://badge.fury.io/py/tensorly

TensorLy
========


Tensor Learning in Python; http://tensorly.github.io


How to install
--------------
 
Easy option: install with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply run::

   pip install -U tensorly

That's it!

Development: install from git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository and cd there::

   git clone https://github.com/tensorly/tensorly
   cd tensorly

Then install the package (here in editable mode with `-e` or equivalently `--editable`)::

   pip install -e .

Alternatively, you can pip install from the git repository::

   pip install git+https://github.com/tensorly/tensorly

Running the tests
~~~~~~~~~~~~~~~~~

Testing is an essential part of this package and all functions come with uni-tests.

You can run all the tests using the `nose` package::

   nosetests -v tensorly

