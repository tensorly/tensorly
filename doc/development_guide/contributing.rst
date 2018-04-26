Contributing
============

All contributions are welcome! So if you have a cool tensor method you want to add, if you spot a bug or even a typo or mistake in the documentation, please report it, and even better, open a Pull-Request!

How-to
------

To make sure the contribution is relevant and is not already worked on, you can `open an issue <https://github.com/tensorly/tensorly/issues>`_ or talk to us on `Gitter <https://gitter.im/tensorly/tensorly>`_!

To add code of fix issues in TensorLy, you will want to open a `Pull-Request <https://github.com/tensorly/tensorly/pulls>`_ on the Github repository of the project. 

Guidelines
----------

For each function or class, we expect helpful docstrings in the NumPy format, as well as unit-tests to make sure it is working as expected (especially helpful for future refactoring to make sure no exising code is broken!)

Check the existing code for examples, and don't hesitate to contact the developers if you are unsure! 


Backend compatibility
---------------------

We want algorithms to run transparently with NumPy, MXNet, PyTorch and any other backend added later on!

To do so, we **only use** functions wrapped in :py:mod:`tensorly.backend`, such as :py:mod:`tensorly.backend.partial_svd`, etc. If the function you need doesn't exist, either try using other existing ones, or, if you cannot do otherwise, add the required function to all backends.

In practice
~~~~~~~~~~~

Practically, **use the wrapped functions**. For instance:

.. code-block:: python

   import tensorly as tl
   import numpy as np
   tensor = tl.tensor(np.random.random((10, 10, 10)))

   # DO THIS:
   min_value = tl.min(tensor)

   # DO NOT DO THIS:
   min_value = np.min(tensor) # Don't do it!


The reason is that you do not want your code to be restricted to any of the backends. 
You might be using NumPy but another user might be using MXNet and calling a NumPy function on an MXNet NDArray will most likely fail.


Context of a tensor
~~~~~~~~~~~~~~~~~~~

An other aspect, when developing a new function or algorithm, is to make sure you perform the computation on the correct context specified by the user. To do so, always get the context from tensors you get as input, and use it for the tensors you create.


.. code-block:: python

   context = tl.context(tensor)
   # when creating a new tensor we use these as parameters
   new_tensor = tl.tensor(tensor + 2, **context)

Check-out the page on :doc:`../user_guide/backend` for more on this.
