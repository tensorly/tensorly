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
         **Notice the space before and after the colon!**

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


Examples
--------
Examples are located in `tensorly/examples` and executed everytime the documentation is built. 
We use the `Sphinx-Gallery <https://sphinx-gallery.github.io/stable/index.html>`_ which you can refer to for documentation but,
in short, your example should be called `plot_xyz.py` and the the structure of that file should typically look like::

   """
   Title of your example
   =====================

   The above title will be the name of the example. The remainder of this text is included in the generated example.

   """

   # Comments are converted into text, while code (below) is executed (and displayed)
   myvariable = 2
   print("my variable is {}".format(myvariable))

   # %%
   # This is a section header
   # ------------------------
   #
   # In the built documentation, it will be rendered as rST. All rST lines
   # must begin with '# ' (note the space) including underlines below section
   # headers.
   #
   # Note
   # ----
   # You can embed rST in your Python examples by including a line of >= 20 # symbols, #%%, or # %%.
   # For consistency, it is recommended that you use only one of the above three ‘block splitter’ options in your project.
   # If using a line of #’s, we recommend using 79 #’s, like this:
   ###############################################################################

