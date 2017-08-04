===================
Installing tensorly
===================

Installing with pip
===================

Simply run::

   pip install tensorly


Cloning the github repository
=============================

Clone the repository and cd there::

   git clone https://github.com/tensorly/tensorly
   cd tensorly

Then install the package (here in editable mode with `-e` or equivalently `--editable`::

   pip install -e .

Running the tests
=================

Uni-testing is an vital part of this package.
You can run all the tests using `pytest`::

   pytest tensorly

