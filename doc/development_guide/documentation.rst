.. _documentation:

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
