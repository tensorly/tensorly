try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

config = {
    'description': 'Tensor learning in Python.',
    'author': 'Jean Kossaifi',
    'author_email': 'jean [dot] kossaifi [at] gmail [dot] com',
    'version': '0.1',
    'install_requires': ['numpy', 'scipy'],
    'packages': find_packages(),
    'scripts': [],
    'name': 'tensorly'
}

setup(**config)
