===================
Installing tensorly
===================


Pre-requisite
=============

The only non-optional pre-requisite is to have Python installed.

.. important::

   TensorLy is developed/tested only for Python3!

   If you are still using Python2, you probably want to upgrade!

If you are starting with Python or generally want a pain-free experience, I recommend you install the `Anaconda distribiution <https://www.anaconda.com/download/>`_. It comes with all you need shipped-in and ready to use!
   

Installing with pip (recommended)
=================================


Simply run, in your terminal::

   pip install -U tensorly

(the `-U` is optional, use it if you want to update the package).


Installing with Conda
=====================

You can also directly install tensorly using conda, from the tensorly channel::

   conda install -c tensorly tensorly

And that is you done! 

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


Why Python 3?
=============

Short answer: it's 2018 (at the time of writing...)

Python 3 was first released **10 years** ago and, while breaking compatibility, it improved a lot of things.

To quote the `wiki <https://wiki.python.org/moin/Python2orPython3>`_:

   | Short version: Python 2.x is legacy, Python 3.x is the present and future of the language

In particular, have a look at the list of `what's new in Python (3) <https://docs.python.org/3/whatsnew/index.html>`_


