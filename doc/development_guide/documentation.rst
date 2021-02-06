.. _documentation:

Writing documentation
=====================

Documentation is a crutial part of this library. Here, we cover some elements of syntax 
with Sphinx and reStructuredText (the `.rst` files), which is the format we use.

Docstrings
----------

All functions and classes should come with useful docstrings. For these, we use the numpy style docstrings. For instance, for a function, we expect the following docstring:

.. code:: python

   def function(arg):
      """One line description
         
      Longer description,
      possibly in several lines

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

Refer to functions or modules
-----------------------------

In the documentation, you can refer to modules as :literal:`:mod:`tensorly.module``, for instance
the tensor algebra module: :mod:`tensorly.tenalg`.
Similarly, use :literal:`:func:`tensorly.function`` to refer a function:
:func:`tensorly.base.unfold`.

Titles
------
Titles are created by underlying them, and the hierarchy is automatically determined by Sphinx.
If you do not want your title to appear in the table of contents, use a `rubric <https://docutils.sourceforge.io/docs/ref/rst/directives.html#rubric>`_::

   .. rubric:: Title
