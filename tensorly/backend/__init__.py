import sys
from .. import _BACKEND

sys.stderr.write('Using {} backend.\n'.format(_BACKEND))

if _BACKEND == 'mxnet':
    from .mxnet_backend import *
elif _BACKEND == 'numpy':
    from .numpy_backend import *
elif _BACKEND == 'pytorch':
    from .pytorch_backend import *
elif _BACKEND == 'tensorflow':
    from .tensorflow_backend import *
elif _BACKEND == 'cupy':
    from .cupy_backend import *
else:
    import warnings
    warnings.warn('_BACKEND should be either "mxnet", "pytorch" or "numpy", {} given.'.format(_BACKEND))
    warnings.warn('Using MXNet backend.')

