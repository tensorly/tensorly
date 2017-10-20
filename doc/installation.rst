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


Compiling the documentation
===========================

You will need to install slimit and minify::

   pip install slimit rcssmin

You are now ready to build the doc (here in html)::

   make html

The results will be in `_build/html`
