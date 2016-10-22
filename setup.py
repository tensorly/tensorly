try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

from tensorly.version import __version__
config = {
    'name': 'tensorly',
    'packages': ['tensorly'],
    'description': 'Tensor learning in Python.',
    'author': 'Jean Kossaifi',
    'author_email': 'jean.kossaifi@gmail.com',
    'version': __version__,
    'url': 'https://github.com/tensorly/tensorly',
    'install_requires': ['numpy', 'scipy'],
    'scripts': []
}

setup(**config)
