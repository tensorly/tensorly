from ..deprecation import deprecated, DefineDeprecated

def test_deprecated():
    class Dummy(object):
        def __init__(self, arg=1):
            self.arg = arg + 1

    def fun1():
        return 2

    # Test using the deprecated function
    @deprecated(Dummy, use_deprecated=True)
    class Deprecated():
        def __init__(self, arg=1):
            self.arg = arg

    @deprecated(fun1, use_deprecated=True)
    def fun2():
        return 1

    instance = Deprecated(1)
    assert instance.arg == 1
    assert fun2() == 1

    # Test using the new function instead
    @deprecated(Dummy, use_deprecated=False)
    class Deprecated():
        def __init__(self, arg=1):
            self.arg = arg

    @deprecated(fun1, use_deprecated=False)
    def fun2():
        return 2

    instance = Deprecated(1)
    assert instance.arg == 2
    assert fun2() == 2
