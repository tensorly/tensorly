=================
Development guide
=================

Contributing
============
All contributions are welcome. 
To make sure the contribution is relevant and is not already work on, you can `open an issue <https://github.com/tensorly/tensorly/issues>`_ or talk to us on `Gitter <https://gitter.im/tensorly/tensorly>`_!

Guidelines
==========
For each function or class, we expect helpful docstrings in the NumPy format, as well as unit-tests to make sure it is working as expected (especially helpful for future refactoring to make sure no exising code is broken!)

Documentation
=============

Documentation is a crutial part of this library.

All functions and classes should come with useful docstrings. For these, we use the numpy style docstrings. For instance, for a function, we expect the following docstring:

.. code:: python

   def function(arg):
      """One line description
         
         Longer description

      Parameters
      ----------
      arg : type
         description

      Returns
      -------
      variable : type
         description

      Examples
      --------
      text
      >>> code
      expected result

      Notes
      -----
      Detailed explanation
      """
      pass



In particular, use single backticks for variable's names: `\`variable\``.

Double backticks are used for inline code: ````inline code````.

For blocks of code, use double colons, leave a white line and indent said lines of code

.. code:: rst

   ::
      
      block of code
      on 
      several 
      lines...
