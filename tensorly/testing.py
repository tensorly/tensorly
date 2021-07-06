import sys
from inspect import getfullargspec

import numpy as np

from tensorly import backend as T


def assert_array_equal(a, b, *args, **kwargs):
    np.testing.assert_array_equal(T.to_numpy(a), T.to_numpy(b),
                                  *args, **kwargs)


def assert_array_almost_equal(a, b, *args, **kwargs):
    np.testing.assert_array_almost_equal(T.to_numpy(a), T.to_numpy(b),
                                         *args, **kwargs)


def assert_equal(actual, desired, *args, **kwargs):
    def _tensor_to_numpy(x):
        if T.is_tensor(x):
            x = T.to_numpy(x)
            return x[0] if x.shape == (1,) else x
        return x

    np.testing.assert_equal(_tensor_to_numpy(actual),
                            _tensor_to_numpy(desired),
                            *args, **kwargs)


def _get_defaultkwargs(func):
    """Returns a dictionary containing all of the input function's arguments with default values.
    """
    argspec = getfullargspec(func)

    arguments = argspec.args
    defaults = argspec.defaults
    kwonlydefaults = argspec.kwonlydefaults
    if defaults is None:
        defaults = tuple()
    if kwonlydefaults is None:
        kwonlydefaults = {}

    start_defaults_idx = len(arguments) - len(defaults)
    arguments = arguments[start_defaults_idx:]
    default_args = {argument: default for argument, default in zip(arguments, defaults)}

    return {**default_args, **kwonlydefaults,}


def _get_decomposition_checker(supposed_kwargs, output_length):
    """Factory function whose output asserts that all entries in ``supposed_kwargs`` match entries in the kwargs-dictionary.

    This is a utility function used to automate testing of the object oriented interface.

    Arguments
    ---------
    supposed_kwargs : dict
        All keyword arguments that should be in the kwargs dict whenever the output function is called
        and their supposed value.
    output_length : int
        The number of outputs from the function
    
    Returns
    -------
    function
        Function that iterates over the supposed_kwarg dictionary and checks that each key and value
        matches those of the function call.
    """
    def decomposition_function(*args, **kwargs):
        for argument, supposed_default in supposed_kwargs.items():
            np.testing.assert_(argument in kwargs, "All arguments with a default must be passed as keyword argument when the decomposition class calls the decomposition function")
            np.testing.assert_(kwargs[argument] == supposed_default)
        return [None for _ in range(output_length)]
    return decomposition_function


def assert_class_wrapper_correctly_passes_arguments(
        monkeypatch, decomposition_function, DecompositionClass, ignore_args=None, decomposition_output_length=2, **extra_args
    ):
    """Used to ensure that all arguments are passed correctly from the decomposition class to the decomposition function

    This code must be used in a test ran with the PyTest framework.

    Arguments:
    ----------
    monkeypatch : pytest.monkeypatch
        Monkeypatch fixture
    decomposition_function : Function
        Decomposition function wrapped by the class
    DecompositionClass : Class
        Class that wraps the function
    ignore_args : iterable
        List of arguments that shouldn't be checked
    decomposition_output_length : int
        Number of outputs from the decomposition function
    **extra_args
        Extra keyword-arguments passed to the decomposition class

    Example:
    --------

    Here is a simple example to check that the CP class' arguments match that of the parafac function.

    >>> from tensorly.decomposition import parafac, CP
    ... def test_cp(monkeypatch):
    ...     assert_class_wrapper_correctly_passes_arguments(monkeypatch, parafac, CP, ignore_args={'return_errors'}, rank=3)
    """
    kwargs = _get_defaultkwargs(decomposition_function)
    test_kwargs = {argument: 'this_is_used_to_test_correct_passing_of_arguments' for argument in kwargs}
    if ignore_args is not None:
        for arg in ignore_args:
            del test_kwargs[arg]
    decomposition_checker = _get_decomposition_checker(test_kwargs, decomposition_output_length)

    decomposition_module = sys.modules[decomposition_function.__module__]
    monkeypatch.setattr(decomposition_module, decomposition_function.__name__, decomposition_checker)
    DecompositionClass(**extra_args, **test_kwargs).fit(None)

    
assert_ = np.testing.assert_
assert_raises = np.testing.assert_raises
