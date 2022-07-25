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

Before you submit your changes, you should also make sure your code adheres to our style-guide. The easiest way to do this is with `black`:  

.. code:: bash

   pip install black
   black .


Check the existing code for examples, and don't hesitate to contact the developers if you are unsure! 


Backend compatibility
---------------------

To contribute code to the TensorLy code-base, you must ensure compatibility with all the backends.

.. important::

   We want algorithms to run transparently with all the TensorLy backends 
   (NumPy, MXNet, PyTorch, TensorLy, JAX, CuPy) and any other backend added later on!

   This means you should only use TensorLy functions, never directly a function from the backend
   e.g. use ``tl.mean``, **not** ``numpy.mean`` or ``torch.mean``.

To do so, we **only use** functions wrapped in :py:mod:`tensorly.backend`, such as :py:mod:`tensorly.backend.tensordot`, etc.
If the function you need doesn't exist, either try using other existing ones,
or, if you cannot do otherwise, add the required function to all backends.

.. important::

   In general, you should **not** use backend specific code, by testing for the backend. 
   e.g. Do not include statements such as ``if tensorly.get_backend() == 'pytorch'`` in your code.


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


Index assignment ("NumPy style")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In NumPy, PyTorch and MXNet, you can combined indexing and assignment in a convenient way, 
e.g. if you have a tensor `t`, you can update its values for given indices using the expression
``t[indices] = values``.

Unfortunately, this is not supported by TensorFlow or JAX. As a result, if you want to do this,
you should use :func:`tensorly.index_update` and :func:`tensorly.index`.
For instance, the previous statement becomes, in TensorLy: 
``t = tensorly.index_update(t, tensorly.index[indices], values)``.


Testing the class interface
---------------------------

Because TensorLy supports both a functional and an object-oriented interface, we should ensure that any
change to one interface is also applied to the other interface. To automatically test for this, we have
a test utility ``tensorly.testing.assert_class_wrapper_correctly_passes_arguments``, which checks that all
keyword arguments of the functional interface are available from the object-oriented interface. 
Here is an example that shows how you can use this utility:


.. code-block:: python

   from tensorly.decomposition import parafac, CP

   def test_cp(monkeypatch):
      assert_class_wrapper_correctly_passes_arguments(monkeypatch, parafac, CP, ignore_args={'return_errors'}, rank=3)

This code will check that all arguments of the ``parafac`` function (except ``return_errors``) can also be
passed to the CP class, and that the input arguments to the CP class are forwarded to the ``parafac`` function.
Notice that the ``test_cp`` function takes an argument: ``monkeypatch``. This lets PyTest know that we
are planning to perform monkeypatching (more about that later), and that it should give us a utility object
for this purpose. It is therefore essential that one of the arguments of the test function is ``monkeypatch``
and that this variable is passed as the first argument to ``assert_class_wrapper_correctly_passes_arguments``.


How the class wrapper test works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This test utility works because it uses the fact that the object-oriented interface calls the functional 
interface behind the scenes. To automatically test the interfaces, the
`tensorly.testing.assert_class_wrapper_correctly_passes_arguments` function inspects the functional interface 
and notes down all possible arguments. Then, it *monkeypatches* the functional interface with a mock function.
In other words, the functional interface is temporarily replaced with a mock function that, instead of fitting
a tensor decomposition, asserts that every possible argument is given a particular placeholder value. 

Since the object-oriented interface uses the functional one behind the scenes, we can then use the object-oriented
interface to fit a model and send in placeholder values for each possible argument of the functional interface.
Then, the monkeypatched code will check that the placeholder value is passed to the functional interface for each argument. 
Finally, the functional interface is reset to normal behaviour (fitting a tensor decomposition). 
