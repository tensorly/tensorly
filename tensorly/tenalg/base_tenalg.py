import warnings

class TenalgBackend():
    _available_tenalg_backends = dict()

    def __init_subclass__(cls, backend_name, **kwargs):
        """When a subclass is created, register it in _known_backends"""
        super().__init_subclass__(**kwargs)

        if backend_name != '':
            cls._available_tenalg_backends[backend_name.lower()] = cls
            cls.backend_name = backend_name
        else:
            warnings.warn(f'Creating a subclass of BaseBackend ({cls.__name__}) with no name.')

    def __repr__(self):
        return f'TensorLy {self.backend_name}-tenalg backend'

    @classmethod
    def register_method(cls, name, func):
        """Register a method with the backend.

        Parameters
        ----------
        name : str
            The method name.
        func : callable
            The method
        """
        setattr(cls, name, staticmethod(func))

