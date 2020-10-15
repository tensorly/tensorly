import warnings


def deprecated(deprecated_by, msg='', use_deprecated=True):
    """Decorator that creates a dummy class or function that returns the class/fun it is deprecated by,
            along with a warning

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/deprecation.py
        but we want to return the new class or function here instead.
    
    Parameters
    ----------
    deprecated_by : function or class
        the new function or class that deprecates the one that is being wrapped
    msg : str, default is ''
        optional message to display in the deprecation warning
    use_deprecated : bool, default is True
        if True, the deprecated wrapped function/class will be used with a warning
        if False, the new class/function will be used along with the deprecation warning for the old name
    """
    class DeprecatedBy(object):
        def __init__(self):
            self.deprecated_by = deprecated_by
            self.use_deprecated = use_deprecated
            self.msg = msg

        def __call__(self, obj):
            if isinstance(obj, type):
                return self._wrap_class(obj)
        
            else:
                return self._wrap_fun(obj)
            
        def _wrap_fun(self, fun):
            def wrapped(*args, **kwargs):
                msg = f'{fun.__name__} is deprecated, use {self.deprecated_by.__name__} instead.'
                if self.msg:
                    msg += '\n' + {self.msg}

                warnings.warn(msg, DeprecationWarning)
                if self.use_deprecated:
                    return fun(*args, **kwargs)
                else:
                    return self.deprecated_by(*args, **kwargs)
            return wrapped
        
        def _wrap_class(self, cls):
            if self.use_deprecated:
                Base = cls
            else:
                Base = self.deprecated_by

            class Wrapped(Base):
                def __init__(wrapped_self, *args, **kwargs):
                    msg = f'{cls.__name__} is deprecated, use {self.deprecated_by.__name__} instead.'
                    if self.msg:
                        msg += '\n' + {self.msg}

                    warnings.warn(msg, DeprecationWarning)
                    super().__init__(*args, **kwargs)
            
            Wrapped.__name__ = cls.__name__
            return Wrapped

    return DeprecatedBy()


class DefineDeprecated(object):
    """Creates a dummy class or function that returns the class/fun it is deprecated by,
            along with a warning
            
        Loosely adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/deprecation.py
            but we want to return the new class or function here instead.

    Example
    -------
    # New function renaming old_fun
    def fun(msg='hello'):
        pass

    # Old fun will return fun but issue a deprecation warning
    old_fun = DefineDeprecated('f', fun)

    >>> f()
    DeprecationWarning: f is deprecated, use fun instead.
    """
    def __new__(cls, deprecated_name, use_instead, msg=''):
        if isinstance(use_instead, type):
            return cls._wrap_class(deprecated_name, use_instead, msg)
        else:
            return cls._wrap_fun(deprecated_name, use_instead, msg)
    
    @classmethod
    def _wrap_fun(cls, deprecated_name, use_instead, msg):
        def wrapped(*args, **kwargs):
            warning = f'{deprecated_name} is deprecated, use {use_instead.__name__} instead.'
            if msg:
                warning += '\n' + {msg}

            warnings.warn(warning, DeprecationWarning)

            return use_instead(*args, **kwargs)
        
        wrapped.__name__ = deprecated_name
        return wrapped
    
    @classmethod
    def _wrap_class(cls, deprecated_name, use_instead, msg):
        class Wrapped(use_instead):
            def __init__(self, *args, **kwargs):
                warning = f'{deprecated_name} is deprecated, use {use_instead.__name__} instead.'
                if msg:
                    warning += '\n' + {msg}

                warnings.warn(warning, DeprecationWarning)
                super().__init__(*args, **kwargs)

        Wrapped.__name__ = deprecated_name
        return Wrapped
